# alphaES

This is the official repository for Alpha Entropy Search for New Information-based Bayesian Optimization. AES generalizes Joint Entropy Search, the former state-of-the-art information-based method for Bayesian Optimization.

#### Status

This work is currently a preprint. For academic use, please refer to the preprint until an official publication is available.

#### Requirements

The following dependencies are required to use this repository:

- botorch >= 0.9.5
- matplotlib >= 3.8.2

#### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/fernandezdaniel/alphaES.git
cd alphaES
pip install -r requirements.txt
```

#### Examples

An example of a 4D experiment for a scenario with noise and another without noise can be found in toy_4D_synthetic_problem/. The methods JES (Joint Entropy Search) (https://github.com/hvarfner/JointEntropySearch), AES with alpha=0.1, and the AES ensemble can be found there.

#### Citation

If you use this repository, please cite:

@article{fernandez2024alpha,
  title={Alpha Entropy Search for New Information-based Bayesian Optimization},
  author={Fern{\'a}ndez-S{\'a}nchez, Daniel and Garrido-Merch{\'a}n, Eduardo C and Hern{\'a}ndez-Lobato, Daniel},
  journal={preprint},
  year={2024}
}

