"""OpenAI Evolution Strategy (Salimans et al. 2017).

[1] https://arxiv.org/abs/1703.03864
[2] https://github.com/hardmaru/estool/blob/master/es.py
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)
from deep_noise_model import DeepNoiseModel, gaussian_log_prob


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState

    noise_log_prob: jax.Array | None = None
    noise_state: Any = None  # TrainState for DeepNoiseModel
    noise_aux_mu: jax.Array | None = None
    noise_aux_std: jax.Array | None = None
    
    # Top-k buffer
    top_k_solutions: jax.Array | None = None
    top_k_fitness: jax.Array | None = None


@struct.dataclass
class Params(BaseParams):
    pass


class VAE_Open_ES(DistributionBasedAlgorithm):
    """OpenAI Evolution Strategy (OpenAI-ES)."""

    def __init__(
            self,
            population_size: int,
            solution: Solution,
            use_antithetic_sampling: bool = False,
            optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
            std_schedule: Callable = optax.constant_schedule(1.0),
            fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
            metrics_fn: Callable = metrics_fn,
            lr_noise_model: float = 1e-4,
            hidden_dims: tuple = (128, 64),
            use_best_individual_augmentation: bool = False,
            alpha=2,
            normalize_fitness_score=False,
            k: int = 200,
    ):

        """Initialize OpenAI-ES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

        # std schedule
        self.std_schedule = std_schedule

        # Antithetic sampling
        self.use_antithetic_sampling = use_antithetic_sampling

        self.deep_noise_model = DeepNoiseModel(
            input_dim=self.num_dims,  # encoder input == num_dims
            latent_dim=self.num_dims,
            hidden_dims=hidden_dims,
            lr=lr_noise_model,
        )

        self.use_best_individual_augmentation = use_best_individual_augmentation
        self.alpha = alpha
        self.normalize_fitness_score = normalize_fitness_score
        self.k = k

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        key, sub = jax.random.split(key)
        noise_state = self.deep_noise_model.init(sub)

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
            noise_state=noise_state,

            noise_log_prob=None,
            top_k_solutions=jnp.zeros((self.k, self.num_dims)),
            top_k_fitness=jnp.full((self.k,), jnp.inf),
        )
        return state

    def _ask(
            self,
            key: jax.Array,
            state: State,
            params: Params,
    ) -> tuple[Population, State]:

        features = state.mean[None, :]

        if self.use_antithetic_sampling:
            pop_half = self.population_size // 2

            z_plus, logp_plus, aux, key = self.deep_noise_model.generate_noise(
                state.noise_state,
                key,
                features=features,
                shape=(pop_half, self.num_dims),
            )

            z = jnp.concatenate([z_plus, -z_plus])
            logp_minus = gaussian_log_prob(aux["mu"], aux["std"], -z_plus)
            logp = jnp.concatenate([logp_plus, logp_minus])  # symmetric
        else:
            z, logp, aux, key = self.deep_noise_model.generate_noise(
                state.noise_state,
                key,
                features=features,
                shape=(self.population_size, self.num_dims),
            )

        population = state.mean + state.std * z
        state = state.replace(noise_log_prob=logp, noise_aux_mu=aux["mu"], noise_aux_std=aux["std"])
        return population, state

    def tell(
            self,
            key: jax.Array,
            population: Population,
            fitness: Fitness,
            state: State,
            params: Params,
    ) -> State:
        # Update top-k buffer using RAW fitness
        # Concatenate current top-k with new population
        all_solutions = jnp.concatenate([state.top_k_solutions, population], axis=0)
        all_fitness = jnp.concatenate([state.top_k_fitness, fitness], axis=0)
        
        # Find top-k indices (smallest fitness)
        # jax.lax.top_k returns largest values, so we negate fitness
        _, top_k_indices = jax.lax.top_k(-all_fitness, self.k)
        
        new_top_k_solutions = all_solutions[top_k_indices]
        new_top_k_fitness = all_fitness[top_k_indices]
        
        state = state.replace(
            top_k_solutions=new_top_k_solutions,
            top_k_fitness=new_top_k_fitness,
        )
        
        return super().tell(key, population, fitness, state, params)

    def add_best_individual_augmentation(self, key, state, population, fitness):
        # Sample from top-k
        # We sample one top-k individual for EACH individual in the population
        pop_size = population.shape[0]
        idx = jax.random.randint(key, shape=(pop_size,), minval=0, maxval=self.k)
        
        best_individual = state.top_k_solutions[idx] # (pop_size, num_dims)
        
        # Use sampled top-k fitness for reward scaling
        sampled_fitness = state.top_k_fitness[idx] # (pop_size,)

        delta_from_best_individual = population - best_individual # (pop_size, num_dims)
        state_vae_mu = state.noise_aux_mu
        state_vae_std = state.noise_aux_std

        if self.use_antithetic_sampling:
            state_vae_mu = jnp.concatenate([state_vae_mu, state_vae_mu])
            state_vae_std = jnp.concatenate([state_vae_std, state_vae_std])

        logprobs_for_noise_delta = gaussian_log_prob(
            state_vae_mu, state_vae_std, delta_from_best_individual
        )
        # reshape best fitness to popsize and scale by alpha
        # sampled_fitness is already (pop_size,)
        shaped_best_fitness = self.alpha * sampled_fitness
        aug_rewards = -shaped_best_fitness
        return aug_rewards, logprobs_for_noise_delta

    def _tell(
            self,
            key: jax.Array,
            population: Population,
            fitness: Fitness,
            state: State,
            params: Params,
    ) -> State:
        # Compute best in population
        new_best = fitness.min()
        best_idx = jnp.argmin(fitness)
        new_best_sol = population[best_idx]

        # Use JAX control flow
        state = jax.lax.cond(
            new_best < state.best_fitness,
            lambda st: st.replace(
                best_fitness=new_best,
                best_solution=new_best_sol,
            ),
            lambda st: st,
            state,
        )

        # Compute grad
        grad = jnp.dot(fitness, (population - state.mean) / state.std) / (
                self.population_size * state.std
        )

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # REINFORCE loss to update noise model
        if self.normalize_fitness_score:
            rewards = -(fitness - fitness.mean())
        else:
            rewards = -fitness

        if not self.use_best_individual_augmentation:
            new_noise_state, _ = self.deep_noise_model.update(
                state.noise_state,
                state.noise_log_prob,
                rewards,
            )
        else:
            rng, key = jax.random.split(key)
            aug_rewards, logprobs_for_noise_delta = self.add_best_individual_augmentation(rng, state, population, fitness)
            cat_rewards = jnp.concatenate([rewards, aug_rewards])
            cat_log_probs = jnp.concatenate([state.noise_log_prob, logprobs_for_noise_delta])
            new_noise_state, _ = self.deep_noise_model.update(
                state.noise_state,
                cat_log_probs,
                cat_rewards,
            )

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
            best_fitness=state.best_fitness,
            generation_counter=state.generation_counter + 1,
            best_solution=state.best_solution,
            noise_state=new_noise_state,
        )
