#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for generalized alpha entropy search (AES).

.. [Hvarfner2022joint]
    C. Hvarfner, F. Hutter, L. Nardi,
    Joint Entropy Search for Maximally-informed Bayesian Optimization.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.

.. [Tu2022joint]
    B. Tu, A. Gandy, N. Kantas, B. Shafei,
    Joint Entropy Search for Multi-objective Bayesian Optimization.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.
"""

from __future__ import annotations

import warnings
from math import log, pi

from typing import Optional

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.model import Model

from botorch.models.utils import check_no_nans, fantasize as fantasize_flag
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from torch.distributions import Normal

MCMC_DIM = -3  # Only relevant if you do Fully Bayesian GPs.

# The CDF query cannot be strictly zero in the division
# and this clamping helps assure that it is always positive.
CLAMP_LB = torch.finfo(torch.float32).eps
FULLY_BAYESIAN_ERROR_MSG = (
    "JES is not yet available with Fully Bayesian GPs. Track the issue, "
    "which regards conditioning on a number of optima on a collection "
    "of models, in detail at https://github.com/pytorch/botorch/issues/1680"
)


class qAlphaEntropySearchEnsemble(AcquisitionFunction, MCSamplerMixin):
    r"""Ensemble of AES with different alpha values.
    Batch is not supported `q = 1`.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        # estimation_type: str = "LB",
        maximize: bool = True,
        num_samples: int = 64,
        alphas: list = None,
        weights_alphas: list = None,
        observation_noise: bool = True,
        eps: float = 1e-6
    ) -> None:
        r"""Alpha entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            X* optimal_inputs: A `num_samples x d`-dim tensor containing the sampled
                optimal inputs of dimension `d`. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            y* optimal_outputs: A `num_samples x 1`-dim Tensor containing the optimal
                set of objectives of dimension `1`.
            condition_noiseless: Whether to condition on noiseless optimal observations
                `f*` [Hvarfner2022joint]_ or noisy optimal observations `y*`
                [Tu2022joint]_. These are sampled identically, so this only controls
                the fashion in which the GP is reshaped as a result of conditioning
                on the optimum.
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC").
                Lower Bound is recommended due to the relatively high variance
                of the MC estimator.
            maximize: If true, we consider a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
            alpha: Hyper-parameter of the acquisition function that generalizes Shannon
            entropy to Alpha entropy. Limit of alpha=1 gives Shannon entropy.
        """
        super().__init__(model=model)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        # To enable fully bayesian GP conditioning, we need to unsqueeze
        # to get num_optima x num_gps unique GPs

        # inputs come as num_optima_per_model x (num_models) x d
        # but we want it four-dimensional in the Fully bayesian case,
        # and three-dimensional otherwise.
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        self.posterior_transform = posterior_transform
        self.maximize = maximize

        # The optima (can be maxima, can be minima) come in as the largest
        # values if we optimize, or the smallest (likely substantially negative)
        # if we minimize. Inside the acquisition function, however, we always
        # want to consider MAX-values. As such, we need to flip them if
        # we want to minimize.
        if not self.maximize:
            optimal_outputs = -optimal_outputs
        self.num_samples = optimal_inputs.shape[0]
        self.condition_noiseless = condition_noiseless
        self.initial_model = model

        # Here, the optimal inputs have shapes num_optima x [num_models if FB] x 1 x D
        # and the optimal outputs have shapes num_optima x [num_models if FB] x 1 x 1
        # The third dimension equaling 1 is required to get one optimum per model,
        # which raises a BotorchTensorDimensionWarning.
        if isinstance(model, SaasFullyBayesianSingleTaskGP):
            raise NotImplementedError(FULLY_BAYESIAN_ERROR_MSG)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with fantasize_flag():
                with settings.propagate_grads(False):
                    # We must do a forward pass one before conditioning.
                    self.initial_model.posterior(
                        self.optimal_inputs[:1], observation_noise=False
                    )

                # This equates to the JES version proposed by Hvarfner et. al.
                if self.condition_noiseless:
                    opt_noise = torch.full_like(
                        self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL
                    )
                    # conditional (batch) model of shape (num_models)
                    # x num_optima_per_model
                    self.conditional_model = (
                        self.initial_model.condition_on_observations(
                            X=self.initial_model.transform_inputs(self.optimal_inputs),
                            Y=self.optimal_outputs,
                            noise=opt_noise,
                        )
                    )
                else:
                    self.conditional_model = (
                        self.initial_model.condition_on_observations(
                            X=self.initial_model.transform_inputs(self.optimal_inputs),
                            Y=self.optimal_outputs,
                        )
                    )

        self.estimation_type = "LB" #estimation_type
        self.set_X_pending(X_pending)

        if alphas is None:
            self.alpha = torch.FloatTensor([ 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999 ])
            self.weights = self.alpha * 0.0 + 1.0
        else:
            self.alpha  = torch.FloatTensor(alphas)
            self.weights_alphas = torch.FloatTensor(weights_alphas)

        print("Ensemble Alpha entropy")



    def g_eta_factor(self, mean, var):
        return 0.5 * torch.log(2 * torch.pi * var) + 0.5 * mean**2 / var

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor, return_parts: bool = False) -> Tensor:
        r"""

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """


        """ Variable mapping 
            self.alpha -> alpha
            -> m2: Conditional mean.
            -> v2: Condicional variance.
            -> m1: Marginal mean y -> mean_m
            -> v1: Marginal variance y -> variance_m
            -> m3: v3 * (alpha * (m2 / v2 - m1 / v1))
            -> v3: 1.0 / (alpha * (1.0 / v2 - 1.0 / v1))
        """

        assert X.shape[ 1 ] == 1, "Batch is not supported yet (q must be 1)"

        #posterior inicial
        initial_posterior = self.initial_model.posterior(X, observation_noise=True)

        mean_pred = initial_posterior.mean.unsqueeze(-1)
        var_pred  = initial_posterior.variance.unsqueeze(-1)

        # need to check if there is a two-dimensional batch shape -
        # the sampled optima appear in the dimension right after
        batch_shape = X.shape[ : -2 ]
        sample_dim = len(batch_shape)

        # Compute the mixture mean and variance
        #media y varianza de la distribucion predictiva.
        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=True
        )

        noiseless_posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False
        )
        noiseless_mean = noiseless_posterior_m.mean
        noiseless_var  = noiseless_posterior_m.variance

        mean_cond = posterior_m.mean
        if not self.maximize:
            mean_cond = -mean_cond
        var_cond = posterior_m.variance

        check_no_nans(var_cond)

        # get stdv of noiseless variance

        noiseless_stdv = noiseless_var.sqrt()

        # batch_shape x 1

        normal = Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        noiseless_normalized_mvs = (self.optimal_outputs - mean_cond) / noiseless_stdv
        noiseless_cdf_mvs = normal.cdf(noiseless_normalized_mvs).clamp_min(CLAMP_LB)
        noiseless_pdf_mvs = torch.exp(normal.log_prob(noiseless_normalized_mvs))

        noiseless_ratio = noiseless_pdf_mvs / noiseless_cdf_mvs

        noiseless_var_cond_trunc = noiseless_var * \
            (1 - (noiseless_normalized_mvs + noiseless_ratio) * noiseless_ratio).clamp_min(CLAMP_LB)

        var_cond_trunc = noiseless_var_cond_trunc + (var_cond - noiseless_var) - 1e-8

        noiseless_mean_cond_trunc = mean_cond - noiseless_stdv * noiseless_ratio
        mean_cond_trunc = noiseless_mean_cond_trunc


        # We add one axis to each variable at the beining

        var_cond_trunc = var_cond_trunc[ None, ...]
        mean_cond_trunc = mean_cond_trunc[ None, ...]
        mean_pred = mean_pred[ None, ...]
        var_pred = var_pred[ None, ...]
        alpha = torch.ones(self.alpha.shape) * self.alpha # Create a copy
        alpha = alpha.reshape([ alpha.shape[ 0 ] ] + [ 1 for i in range(len(var_cond_trunc.shape) - 1)])

        v3 = 1.0 / (alpha * (1.0 / var_cond_trunc - 1.0 / var_pred))
        m3 = v3 * (alpha * (mean_cond_trunc / var_cond_trunc - mean_pred / var_pred))

        # We prepare the values to compute the integral analitically

        g_m1v1 = self.g_eta_factor(mean_pred, var_pred)
        g_m2v2 = self.g_eta_factor(mean_cond_trunc, var_cond_trunc)
        g_m3v3 = self.g_eta_factor(m3, v3)

        v_prod_pred_dist_3 = 1.0 / (1.0 / v3 + 1.0 / var_pred)
        m_prod_pred_dist_3 = v_prod_pred_dist_3 * (m3 / v3 + mean_pred / var_pred)

        # We compute the integral: int (p(y|x*) / p(y))^alpha p(y) d y

        #res = torch.exp(- self.alpha * g_m2v2 + self.alpha * g_m1v1 + g_m3v3) * torch_pdf(mean_pred, m3, var_pred + v3)
        res = torch.exp(- alpha * g_m2v2 + alpha * g_m1v1 - g_m1v1 + self.g_eta_factor( m_prod_pred_dist_3, v_prod_pred_dist_3))

        # XXX Checked. This last squeeze in 2-D problems

        res = ((1.0 / ( alpha * ( 1.0 - alpha ))).mean(dim=sample_dim + 1).squeeze(-1).squeeze(-1) - \
                (1.0 / (alpha * (1.0 - alpha ))).mean(dim=sample_dim + 1).squeeze(-1).squeeze(-1) * \
                res.mean(dim=sample_dim + 1).squeeze(-1).squeeze(-1))
        
        # We divide the acquisition by the weights_alphas 

        return ( res / abs(self.weights_alphas[ :, None ]) ).mean(0) # We average across alpha values # To take into account p(x*) we use MC samples 



