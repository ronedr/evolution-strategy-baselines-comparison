import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from vae_flax import VAEEncoder


def gaussian_log_prob(mu, std, x):
    var = std ** 2 + 1e-8
    return -0.5 * (jnp.log(2 * jnp.pi * var) + (x - mu) ** 2 / var).sum(axis=-1)


class DeepNoiseModel:
    def __init__(
            self,
            rng_key,
            input_dim,
            hidden_dims=(128, 64),
            lr=1e-4,
    ):
        self.rng = rng_key

        # --- build encoder
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=input_dim,
            hidden_dims=hidden_dims,
        )

        # init params
        dummy = jnp.zeros((1, input_dim))
        variables = self.encoder.init(self.rng, dummy)
        self.state = TrainState.create(
            apply_fn=self.encoder.apply,
            params=variables["params"],
            tx=optax.adam(lr),
        )

    # -------------------------------------------------------
    # Noise generation (replaces OpenES random sampling step)
    # -------------------------------------------------------
    def generate_noise(self, rng, features, shape):
        """
        features: jnp.ndarray shape (batch, input_dim)
        shape: (population_size, num_dims)

        returns:
            noise: (pop, dims)
            log_prob: (pop,)
            aux: a dict storing mu,std,eps useful for training
        """
        mu, std, _ = self.state.apply_fn({"params": self.state.params}, features)

        # broadcast to match population shape
        mu = jnp.broadcast_to(mu, shape)
        std = jnp.broadcast_to(std, shape)

        rng, sub = jax.random.split(rng)
        eps = jax.random.normal(sub, shape)

        noise = mu + std * eps

        log_prob = gaussian_log_prob(mu, std, noise)

        aux = {"mu": mu, "std": std, "eps": eps}
        return noise, log_prob, aux, rng

    # -------------------------------------------------------
    # REINFORCE update
    # -------------------------------------------------------
    @jax.jit
    def reinforce_step(self, state, log_prob, rewards):
        """
        REINFORCE loss = -(reward * log_prob)
        log_prob and reward shapes: (population_size,)
        """

        def loss_fn(params):
            loss = -(log_prob * rewards).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def update(self, log_prob, rewards):
        self.state, loss = self.reinforce_step(self.state, log_prob, rewards)
        return loss
