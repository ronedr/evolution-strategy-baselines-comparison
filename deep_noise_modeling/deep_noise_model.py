import functools

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from deep_noise_modeling.vae_unet import VAEUNetEncoder
from vae_flax import VAEEncoder
from srht_random_projection import SRHT_Projection_Padded
from evosax.algorithms import EvoTF_ES, Open_ES


def gaussian_log_prob(mu, std, x):
    var = std ** 2 + 1e-8
    return -0.5 * (jnp.log(2 * jnp.pi * var) + (x - mu) ** 2 / var).sum(axis=-1)


class DeepNoiseModel:
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dims=(128, 64),
            lr=1e-4,
            use_random_projection=False,
            random_projection_dim=None,
    ):
        if use_random_projection:
            assert random_projection_dim is not None, "random_projection_dim must be specified"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.use_random_projection = use_random_projection
        self.random_projection_dim = random_projection_dim
        self.projection = SRHT_Projection_Padded(input_dim=input_dim,
                                                 output_dim=random_projection_dim) if use_random_projection else None
        self.proj_params = None
        # self.encoder = VAEEncoder(
        #     input_dim=input_dim if not use_random_projection else random_projection_dim,
        #     latent_dim=latent_dim,
        #     hidden_dims=hidden_dims,
        # )

        self.encoder = VAEUNetEncoder(input_dim=input_dim if not use_random_projection else random_projection_dim,
                                      latent_dim=latent_dim)

        self.tx = optax.adam(lr)

    @functools.partial(jax.jit, static_argnames=("self",))
    def init(self, rng):
        """Returns TrainState, with no Python-side mutation."""
        dummy = jnp.zeros((1, self.input_dim if not self.use_random_projection else self.random_projection_dim))
        variables = self.encoder.init(rng, dummy)
        return TrainState.create(
            apply_fn=self.encoder.apply,
            params=variables["params"],
            tx=self.tx,
        )

    @functools.partial(jax.jit, static_argnames=("self", "shape"))
    def generate_noise(self, noise_state, rng, features, shape):
        """
        noise_state: TrainState (carried in ES.State)
        features: (batch, input_dim)
        shape: (popsize, num_dims)
        """
        if self.proj_params is None and self.use_random_projection:
            self.proj_params = self.projection.init(rng, features)

        if self.use_random_projection:
            features = self.projection.apply(self.proj_params, features)

        mu, std, _ = noise_state.apply_fn({"params": noise_state.params}, features)

        mu = jnp.broadcast_to(mu, shape)
        std = jnp.broadcast_to(std, shape)

        rng, sub = jax.random.split(rng)
        eps = jax.random.normal(sub, shape)

        noise = mu + std * eps
        log_prob = gaussian_log_prob(mu, std, noise)

        aux = {"mu": mu, "std": std}
        return noise, log_prob, aux, rng

    # -------------------------------------------------------
    # REINFORCE update
    # -------------------------------------------------------
    @staticmethod
    @jax.jit
    def reinforce_step(noise_state, log_prob, rewards):
        """Pure function: returns new TrainState + loss."""

        def loss_fn(params):
            return -(log_prob * rewards).mean()

        loss, grads = jax.value_and_grad(loss_fn)(noise_state.params)
        new_state = noise_state.apply_gradients(grads=grads)
        return new_state, loss

    @functools.partial(jax.jit, static_argnames=("self",))
    def update(self, noise_state, log_prob, rewards):
        return DeepNoiseModel.reinforce_step(noise_state, log_prob, rewards)
