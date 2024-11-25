import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from botorch.utils.dispatcher import Dispatcher
from typing import Any, Callable, Iterable, List, Optional, overload, Tuple, Union
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from botorch.sampling.pathwise.utils import GetTrainInputs

# Define a GP model with Matern kernel and no noise

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, num_dims, likelihood_noise: float = 1e-4, lengthscale: float = 0.25):
        
        self.num_dims = num_dims
        self.likelihood_noise = likelihood_noise
        self.lengthscale = lengthscale

        # We prepare the Likelihood and specify its noise

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.likelihood_noise

        # Correctly set the lengthscale for the dimensions. Use default values for other parameters (e.g. amplitude)

        super(ExactGPModel, self).__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.num_dims))
        self.covar_module.base_kernel.lengthscale = torch.ones(self.num_dims) * self.num_dims * self.lengthscale
        self.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Hack to know what are the training inputs. It is just fake data, but required by draw_kernel_feature_paths to know the
# input dimensionality.

@GetTrainInputs.register(ExactGPModel)
def _get_train_inputs_Model(model: ExactGPModel, transformed: bool = False) -> Tuple[Tensor]:
    return torch.ones((1, model.num_dims))

class Synthetic_problem():

    def __init__(self, num_dims, lengthscale_model: float = 0.25, seed=None):

        # if os.path.isfile('problem.dat'): # XXX DFS: In future version it will be nice to have something to load the last model
        #     problem = load_object('problem.dat')
        #     self.tasks = problem.tasks
        #     self.input_space = problem.input_space
        #     self.models = problem.models
        #     self.funs = problem.funs
        #     return

        self.num_dims = num_dims
        self.lengthscale_model = lengthscale_model
        self.seed = seed

        # We get the current random state of numpy and torch
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        # We set our random state
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize the model 

        model = ExactGPModel(num_dims, lengthscale=self.lengthscale_model)
        model.double()

        # Draw function

        self.paths = draw_kernel_feature_paths(model, sample_shape=torch.Size([1]))

        # We restore the random state
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    def _plot_synthetic_problem_1D(self, resolution):

        self.x = torch.linspace(0, 1, resolution)

        # Plot the samples
        plt.figure(figsize=(16, 12))
        plt.plot(self.x, self.paths(self.x))
        plt.title('Sample from the GP prior with Matern Kernel')
        plt.colorbar()
        plt.show()

    def _plot_synthetic_problem_2D(self, resolution):

        x1 = torch.linspace(0, 1, self.resolution)
        x2 = torch.linspace(0, 1, self.resolution)
        x1, x2 = torch.meshgrid(x1, x2)
        x = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
           # sampled_y = model(x).sample(sample_shape=torch.Size([1]))
            sampled_y = self.f(x)

        # Plot the samples
        plt.figure(figsize=(16, 12))
        plt.contourf(x1.numpy(), x2.numpy(), sampled_y[0].reshape(50, 50).numpy())
        plt.title('Sample from the GP prior with Matern Kernel')
        plt.colorbar()
        plt.show()

    def plot_synthetic_problem(self, resolution):

        if self.num_dims == 1: self._plot_synthetic_problem_1D(self, resolution)

        if self.num_dims == 2: self._plot_synthetic_problem_2D(self, resolution)

        raise ValueError(f"Only 1D and 2D samples is supported for plotting. The dimensionality of your problem: {self.num_dims}.")
        
    def f(self, x):

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            evaluation = self.paths(x)

        return evaluation

    def f_noisy(self, x):

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            evaluation = self.paths(x) + np.random.normal() * np.sqrt(1.0 / 100)

        return evaluation

