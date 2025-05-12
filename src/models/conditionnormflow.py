import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN, ConditionalAutoRegressiveNN


class ConditionalNormalizingFlow(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        flow_length,
        count_bins,
        bound,
        device=None,
        hidden_dim=1024,
    ):
        super(ConditionalNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.flow_length = flow_length
        self.count_bins = count_bins
        # self.order = order
        self.bound = bound
        self.device = device
        # self.device = "cpu" if not use_cuda else "cuda"

        self.has_prop_score = False
        self.cond_base_dist = dist.MultivariateNormal(
            torch.zeros(self.input_dim).float().to(self.device),
            torch.diag(torch.ones(self.input_dim)).float().to(self.device),
        )  # .to(self.device)

        self.cond_loc = torch.nn.Parameter(torch.zeros((self.input_dim,)).float()).to(
            self.device
        )
        self.cond_scale = torch.nn.Parameter(torch.ones((self.input_dim,)).float()).to(
            self.device
        )
        self.cond_affine_transform = T.AffineTransform(
            self.cond_loc, self.cond_scale
        )  # .to(self.device)

        assert self.input_dim > 1

        self.cond_spline_nn = (
            ConditionalAutoRegressiveNN(
                self.input_dim,
                self.context_dim,
                [self.hidden_dim],
                param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)],
            )
            .float()
            .to(self.device)
        )
        self.cond_spline_transform = [
            T.ConditionalSplineAutoregressive(
                self.input_dim,
                self.cond_spline_nn,
                order="quadratic",
                count_bins=self.count_bins,
                bound=self.bound,
            ).to(self.device)
            for _ in range(self.flow_length)
        ]

        # flow network
        self.flow_dist = dist.ConditionalTransformedDistribution(
            self.cond_base_dist,
            [self.cond_affine_transform] + self.cond_spline_transform,
        )  # .to( self.device)  # [self.cond_affine_transform, self.cond_spline_transform]

    def sample(self, H, num_samples=1):
        assert num_samples >= 1
        num_H = H.shape[0] if len(H.shape) == 2 else 1
        dim_samples = (
            [num_samples, num_H]
            if (num_samples > 1 and num_H > 1)
            else [num_H]
            if num_H > 1
            else [num_samples]
        )
        x = self.flow_dist.condition(H).sample(dim_samples)
        return x

    def log_prob(self, x, H):
        # x = x.reshape(-1, self.input_dim)
        cond_flow_dist = self.flow_dist.condition(H)  # .to(self.device)
        # print(x.shape, H.shape)
        return cond_flow_dist.log_prob(x)

    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.cond_spline_transform))
        with pyro.plate("data", N):
            self.cond_flow_dist = self.flow_dist.condition(H)
            obs = pyro.sample("obs", self.cond_flow_dist, obs=X)

    def guide(self, X=None, H=None):
        pass
