import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll

def fit_gp_models(X, Y, kernel_fn=None, noise_prior=None):
    """
    Fits a list of GP models for each output dimension.
    Returns a ModelListGP.

    Args:
        X: torch tensor of shape (N, D)
        Y: torch tensor of shape (N, M)
        kernel_fn: function taking D and returning a GPyTorch kernel
        noise_prior: noise prior to use in the likelihood

    Returns:
        model: ModelListGP with one GP per output
    """
    models = []
    for j in range(Y.shape[1]):
        train_y = Y[:, j].unsqueeze(-1)
        model = SingleTaskGP(X, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        models.append(model)

    return ModelListGP(*models)