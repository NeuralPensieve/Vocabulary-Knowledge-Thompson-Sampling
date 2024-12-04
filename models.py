import tqdm
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import AffineTransform, SigmoidTransform
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import numpy as np
from scipy.stats import norm, truncnorm
import torch


class SimulatedUserEnv:
    def __init__(self, true_mu: float, true_std: float):
        """
        Initialize the simulated environment.

        Args:
            true_mu (float): The mean of the user's knowledge level.
            true_std (float): The standard deviation of the user's knowledge level.
        """
        self.true_mu = true_mu
        self.true_std = true_std

    def query(self, word_difficulty: float) -> int:
        """
        Query whether the user knows the word.

        Args:
            word_difficulty (float): The difficulty of the word. Higher means harder.

        Returns:
            int: 1 if the user knows the word, 0 otherwise.
        """
        prob_knows = 1 - norm.cdf(
            word_difficulty, loc=self.true_mu, scale=self.true_std
        )
        return int(np.random.rand() < prob_knows)


class VocabularyKnowledgeModel:
    def __init__(
        self, mu0=12_500, std0=7_000, max_num_words=25_000, opt_iterations=5_000
    ):
        """
        Initialize the vocabulary knowledge model.

        Args:
            mu0 (float): Initial mean estimate of user's knowledge level
            std0 (float): Initial standard deviation estimate
            max_num_words (int): Maximum number of words in the vocabulary
        """
        self.mu0 = mu0
        self.std0 = std0
        self.max_num_words = max_num_words
        self.history_mu = [mu0]
        self.history_std = [std0]
        self.lower_bound = 0
        self.upper_bound = max_num_words
        self.observed_difficulties = []
        self.observed_responses = []
        self.opt_iterations = opt_iterations

        self.sigmoid_to_bounded = dist.transforms.ComposeTransform(
            [
                SigmoidTransform(),
                AffineTransform(
                    loc=self.lower_bound, scale=self.upper_bound - self.lower_bound
                ),
            ]
        )

    def sample_from_prior(self):
        """
        Sample from the truncated prior distribution.
        Returns a value between 0 and max_num_words.
        """
        a = (self.lower_bound - self.mu0) / self.std0
        b = (self.upper_bound - self.mu0) / self.std0
        sample = truncnorm.rvs(a, b, loc=self.mu0, scale=self.std0)
        return sample

    def update_posterior(self, x_data, y_data):
        """Update the model's beliefs based on new observations."""
        # Properly handle tensor conversion
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)
        else:
            x_data = x_data.clone().detach().float()

        if not isinstance(y_data, torch.Tensor):
            y_data = torch.tensor(y_data, dtype=torch.float32)
        else:
            y_data = y_data.clone().detach().float()

        def model(x, y):
            # we first sample from unconstrained N(0, 1), and then apply sigmoid, and then scale it
            mu_unconstrained = pyro.sample("mu_unconstrained", dist.Normal(0, 1))
            mu = self.sigmoid_to_bounded(mu_unconstrained)

            # std os sampled from HalfNormal
            std = pyro.sample("std", dist.HalfNormal(5000.0))

            logits = -(x - mu) / std
            probs = torch.sigmoid(logits)

            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Bernoulli(probs), obs=y)

        def guide(x, y):
            init_mu_loc = self.sigmoid_to_bounded.inv(torch.tensor(self.mu0))
            mu_loc = pyro.param("mu_loc", init_mu_loc)
            mu_scale = pyro.param(
                "mu_scale", torch.tensor(1.0), constraint=dist.constraints.positive
            )
            mu_unconstrained = pyro.sample(
                "mu_unconstrained", dist.Normal(mu_loc, mu_scale)
            )

            std_loc = pyro.param(
                "std_loc", torch.tensor(self.std0), constraint=dist.constraints.positive
            )
            pyro.sample("std", dist.HalfNormal(std_loc))

        # Rest of the method remains the same
        optimizer = Adam({"lr": 0.001})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        pbar = tqdm.trange(self.opt_iterations, desc="Training", leave=True)
        for step in pbar:
            loss = svi.step(x_data, y_data)
            if step % 200 == 0:
                pbar.set_postfix({"loss": f"{loss:.4f}"})

        mu_unconstrained = pyro.param("mu_loc").item()
        mu_constrained = self.sigmoid_to_bounded(torch.tensor(mu_unconstrained)).item()
        std = pyro.param("std_loc").item()

        self.mu0 = mu_constrained
        self.std0 = std

        self.history_mu.append(mu_constrained)
        self.history_std.append(std)

    def thompson_sampling(self, env, num_iterations=50):
        """Perform Thompson sampling to efficiently estimate user's knowledge level."""
        x_data = []
        y_data = []

        for itr in range(num_iterations):
            sampled_difficulty = self.sample_from_prior()
            response = env.query(sampled_difficulty)

            x_data.append(sampled_difficulty)
            y_data.append(response)
            self.observed_difficulties.append(sampled_difficulty)
            self.observed_responses.append(response)

            # No need to convert to tensor here since update_posterior handles it
            self.update_posterior(x_data, y_data)

            print("*" * 30)
            print(f"iteration: {itr}")
            print(f"difficulty: {sampled_difficulty:.0f}. Response: {response}.")
            print(f"mu: {self.history_mu[-1]:.0f}. std: {self.history_std[-1]:.0f}.")
            print("*" * 30)