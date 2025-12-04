
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from typing import Any, Callable, Optional, Tuple
from evosax.algorithms.base import State as BaseState
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    metrics_fn,
)
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution
from evosax.algorithms.distribution_based.evotf_es import EvoTransformer
from deep_noise_model import gaussian_log_prob
from flax.training.train_state import TrainState
import functools

# --------------------------------------------------------------------------------
# EvoTF Noise Model (Wraps EvoTransformer + Projection Head + PPO)
# --------------------------------------------------------------------------------

class EvoTFNoiseModel(nn.Module):
    input_dim: int
    output_pop_size: int # Number of noise vectors to generate (e.g. pop_size or pop_size//2)
    model_config: dict

    def setup(self):
        # EvoTransformer core
        self.transformer = EvoTransformer(
            embed_dim=self.model_config.get("embed_dim", 64),
            num_heads=self.model_config.get("num_heads", 4),
            num_latents=self.model_config.get("num_latents", 16),
            latent_dim=self.model_config.get("latent_dim", 32),
            num_layers=self.model_config.get("num_layers", 2),
            use_fitness_encoder=False, # Debug: disable
            use_dist_encoder=False, # Debug: disable
            use_crossd_encoder=False,
        )

    def __call__(self, solution_features, fitness_features, dist_features):
        # x: (batch, seq_len, dims)
        # f: (batch, seq_len, 1)
        # d: (batch, seq_len, dims)
        
        batch_size, seq_len, dims = solution_features.shape
        
        # Revert to 5D inputs
        # x_in: (batch, seq_len, pop_size=1, num_dims=dims, feature_dim=1)
        x_in = solution_features.reshape(batch_size, seq_len, 1, dims, 1)
        
        # f_in: (batch, seq_len, pop_size=1, feature_dim=1)
        f_in = fitness_features.reshape(batch_size, seq_len, 1, 1)
        
        # d_in: (batch, seq_len, pop_size=1, feature_dim=dims) -> (batch, seq_len, dims)
        d_in = dist_features.reshape(batch_size, seq_len, dims)
        
        out = self.transformer(x_in, f_in, d_in)
        
        # Handle tuple return
        if isinstance(out, tuple):
            distrib_out = out[0] 
        else:
            distrib_out = out
            
        # distrib_out shape: (2, batch, seq_len, num_dims)
        # We want (batch, num_dims, 2) -> mu, sigma
        
        # Mean pool over sequence length (axis 2)
        distrib_out = jnp.mean(distrib_out, axis=2) # (2, batch, num_dims)
        
        # Transpose to (batch, num_dims, 2)
        distrib_out = distrib_out.transpose(1, 2, 0)
        
        mu = distrib_out[..., 0] # (batch, num_dims)
        log_std = distrib_out[..., 1] # (batch, num_dims)
        
        # Broadcast to output_pop_size
        mu = jnp.expand_dims(mu, axis=1) # (batch, 1, num_dims)
        mu = jnp.broadcast_to(mu, (batch_size, self.output_pop_size, self.input_dim))
        
        log_std = jnp.expand_dims(log_std, axis=1)
        log_std = jnp.broadcast_to(log_std, (batch_size, self.output_pop_size, self.input_dim))
        
        # Bound sigma to be reasonable
        sigma = jnp.exp(jnp.clip(log_std, -5.0, 2.0))
        
        return mu, sigma

class EvoTFNoiseModelWrapper:
    def __init__(self, input_dim, output_pop_size, model_config, lr=3e-4, ppo_clip=0.2, ppo_epochs=4):
        self.input_dim = input_dim
        self.output_pop_size = output_pop_size
        self.model_config = model_config
        self.lr = lr
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        
        self.model = EvoTFNoiseModel(
            input_dim=input_dim,
            output_pop_size=output_pop_size,
            model_config=model_config
        )
        
        self.tx = optax.adam(lr)

    @functools.partial(jax.jit, static_argnames=("self",))
    def init(self, rng):
        # Create dummy inputs for init
        dummy_x = jnp.zeros((1, 10, self.input_dim))
        dummy_f = jnp.zeros((1, 10, 1))
        dummy_d = jnp.zeros((1, 10, self.input_dim))
        
        variables = self.model.init(rng, dummy_x, dummy_f, dummy_d)
        
        train_state = TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=self.tx,
        )
        return train_state

    @functools.partial(jax.jit, static_argnames=("self", "shape"))
    def generate_noise(self, noise_state, rng, solution_hist, fitness_hist, dist_hist, shape):
        """
        Generate noise using the EvoTF model.
        """
        mu, sigma = noise_state.apply_fn(
            {"params": noise_state.params},
            solution_hist, fitness_hist, dist_hist
        )
        
        # mu, sigma are (batch, output_pop_size, dims)
        # We assume batch=1
        mu = mu[0]
        sigma = sigma[0]
        
        rng, sub = jax.random.split(rng)
        eps = jax.random.normal(sub, shape)
        
        noise = mu + sigma * eps
        log_prob = gaussian_log_prob(mu, sigma, noise)
        
        aux = {"mu": mu, "std": sigma}
        return noise, log_prob, aux, rng

    # -------------------------------------------------------
    # PPO Update
    # -------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=("self",))
    def update(self, noise_state, rng, solution_hist, fitness_hist, dist_hist, old_log_probs, advantages, actions):
        """
        PPO update step.
        """
        
        def loss_fn(params):
            mu, sigma = self.model.apply({"params": params}, solution_hist, fitness_hist, dist_hist)
            mu = mu[0] # (output_pop, dims)
            sigma = sigma[0] # (output_pop, dims)
            
            # Calculate new log probs for the actions taken
            # actions: (output_pop, dims)
            new_lp = gaussian_log_prob(mu, sigma, actions) # (output_pop,)
            
            ratio = jnp.exp(new_lp - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
            
            loss = -jnp.minimum(surr1, surr2).mean()
            return loss

        def train_step(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return (new_state, key), loss

        (new_state, _), losses = jax.lax.scan(
            train_step, (noise_state, rng), None, length=self.ppo_epochs
        )
        
        return new_state, losses.mean()


# --------------------------------------------------------------------------------
# EvoTF OpenES Algorithm
# --------------------------------------------------------------------------------

@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState
    
    # Noise model state
    noise_state: TrainState
    noise_log_prob: jax.Array
    noise_aux_mu: jax.Array
    noise_aux_std: jax.Array
    
    # History buffers for EvoTF
    sol_history: jax.Array
    fit_history: jax.Array
    dist_history: jax.Array
    
    # We don't strictly need a pointer if we just roll the buffer
    # But let's just use roll.


@struct.dataclass
class Params(BaseParams):
    pass


class EvoTF_Open_ES(DistributionBasedAlgorithm):
    """EvoTF OpenES with PPO-optimized noise model."""

    def __init__(
            self,
            population_size: int,
            solution: Solution,
            num_dims: int,
            context_len: int = 10,
            use_antithetic_sampling: bool = True,
            optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
            std_schedule: Callable = optax.constant_schedule(1.0),
            fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
            metrics_fn_: Callable = metrics_fn,
            model_config: dict = None,
            noise_model_lr: float = 3e-4,
            ppo_epochs: int = 4,
            normalize_fitness_score: bool = True,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn_)
        self.num_dims = num_dims
        self.context_len = context_len
        self.use_antithetic_sampling = use_antithetic_sampling
        self.optimizer = optimizer
        self.std_schedule = std_schedule
        self.normalize_fitness_score = normalize_fitness_score
        
        if model_config is None:
            model_config = {}
        
        # Determine output population size for the noise model
        self.eff_pop_size = population_size // 2 if use_antithetic_sampling else population_size
            
        self.noise_model = EvoTFNoiseModelWrapper(
            input_dim=num_dims,
            output_pop_size=self.eff_pop_size,
            model_config=model_config,
            lr=noise_model_lr,
            ppo_epochs=ppo_epochs
        )

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        key, sub = jax.random.split(key)
        noise_state = self.noise_model.init(sub)
        
        state = State(
            mean=jnp.zeros(self.num_dims),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
            
            noise_state=noise_state,
            noise_log_prob=jnp.zeros((self.eff_pop_size,)),
            noise_aux_mu=jnp.zeros((self.eff_pop_size, self.num_dims)),
            noise_aux_std=jnp.ones((self.eff_pop_size, self.num_dims)),
            
            sol_history=jnp.zeros((1, self.context_len, self.num_dims)),
            fit_history=jnp.zeros((1, self.context_len, 1)),
            dist_history=jnp.zeros((1, self.context_len, self.num_dims)),
        )
        return state

    def _ask(
            self,
            key: jax.Array,
            state: State,
            params: Params,
    ) -> Tuple[Population, State]:
        
        # Generate noise from features (history)
        z, logp, aux, key = self.noise_model.generate_noise(
            state.noise_state,
            key,
            state.sol_history,
            state.fit_history,
            state.dist_history,
            shape=(self.eff_pop_size, self.num_dims)
        )
        
        if self.use_antithetic_sampling:
            z_full = jnp.concatenate([z, -z])
            # For antithetic, we don't really have a log prob for the second half in the same way
            # But for PPO update, we only update based on the generated noise 'z'.
            # The second half is deterministic given z.
            # So we track logp for 'z'.
        else:
            z_full = z
            
        population = state.mean + z_full * state.std # Apply global std scaling on top?
        # Or should the model predict the full noise?
        # Standard ES: theta + sigma * epsilon.
        # Here z is our epsilon (but shaped by mu/sigma from model).
        # So yes, we scale by global std.
        
        state = state.replace(
            noise_log_prob=logp,
            noise_aux_mu=aux["mu"],
            noise_aux_std=aux["std"]
        )
        return population, state

    def _tell(
            self,
            key: jax.Array,
            population: Population,
            fitness: Fitness,
            state: State,
            params: Params,
    ) -> State:
        
        # Standard ES Update
        # ------------------
        
        # Reconstruct noise
        # If antithetic: population = [mean+z, mean-z] (scaled by std)
        # z = (pop - mean) / std
        
        # We need the 'z' that was generated by the model to compute advantages for PPO.
        # In _ask, we generated 'z' (eff_pop_size).
        # If antithetic, the first half of population corresponds to +z.
        
        z_full = (population - state.mean) / state.std
        
        if self.use_antithetic_sampling:
            z_model = z_full[:self.eff_pop_size]
            # Fitness for z_model?
            # We have fitness for +z and -z.
            # We can average them or treat them as separate samples?
            # But PPO update expects 'actions' and 'advantages'.
            # Our action was 'z'.
            # The return associated with 'z' is ... ?
            # In standard ES, gradient is sum(F_i * eps_i).
            # For PPO, we want to increase prob of 'z' if it yields high return.
            # If antithetic, we have F(+z) and F(-z).
            # We can define reward for 'z' as (F(+z) - F(-z))/2 ? Or max?
            # Or just use F(+z)?
            # Let's use the average performance or similar.
            # Actually, standard antithetic gradient is (Fpos - Fneg) * eps.
            # So the "advantage" of eps is (Fpos - Fneg).
            
            f_pos = fitness[:self.eff_pop_size]
            f_neg = fitness[self.eff_pop_size:]
            
            # If minimizing fitness, lower is better.
            # We want to maximize reward.
            # Reward = -Fitness.
            # Advantage ~ (R_pos - R_neg) ?
            # If R_pos > R_neg, z was good.
            # If R_pos < R_neg, z was bad (should have been -z).
            
            # Let's define advantage = -(f_pos - f_neg) / 2
            # If f_pos is small (good) and f_neg is large (bad), advantage is positive.
            
            raw_advantages = -(f_pos - f_neg)
            
        else:
            z_model = z_full
            raw_advantages = -fitness
            
        # Normalize advantages
        if self.normalize_fitness_score:
            advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
        else:
            advantages = raw_advantages

        # Update Mean
        # -----------
        # Standard ES gradient
        # grad = 1/(N*sigma) * sum(F_i * eps_i)
        # eps_i here is z_full_i
        
        # We can use the standard update logic or the one from VAE_Open_ES
        # VAE_Open_ES uses "whitened" epsilon based on model predicted mu/sigma.
        # eps_whitened = (eps - mu) / sigma^2
        # But here 'z_full' IS the noise sample from the model's perspective (before global std).
        # Wait, in _ask: population = mean + z_full * state.std
        # So z_full is the "epsilon" in standard ES terms.
        # But z_full ~ N(mu_model, sigma_model).
        
        # If we want to follow VAE_Open_ES logic:
        # eps_centered = z_model - noise_mu
        # eps_whitened = eps_centered / (noise_sigma ** 2)
        # grad = dot(fitness, eps_whitened) ...
        
        # But wait, we have z_full (all samples).
        # We need to reconstruct full mu/sigma for z_full.
        
        noise_mu = state.noise_aux_mu
        noise_sigma = state.noise_aux_std
        
        if self.use_antithetic_sampling:
            # Mirror for negative samples?
            # If z ~ N(mu, sigma), then -z ~ N(-mu, sigma).
            noise_mu_full = jnp.concatenate([noise_mu, -noise_mu])
            noise_sigma_full = jnp.concatenate([noise_sigma, noise_sigma])
        else:
            noise_mu_full = noise_mu
            noise_sigma_full = noise_sigma
            
        eps_centered = z_full - noise_mu_full
        eps_whitened = eps_centered / (noise_sigma_full ** 2 + 1e-8)
        
        # Gradient computation
        # grad = sum(F * eps_whitened) / (pop * global_std)
        grad = jnp.dot(fitness, eps_whitened) / (self.population_size * state.std)
        
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)
        
        # Update Noise Model (PPO)
        # ------------------------
        # We update based on z_model, old_log_probs, and advantages
        
        rng, key = jax.random.split(key)
        new_noise_state, ppo_loss = self.noise_model.update(
            state.noise_state,
            rng,
            state.sol_history,
            state.fit_history,
            state.dist_history,
            state.noise_log_prob,
            advantages,
            z_model
        )
        
        # Update History
        # --------------
        # We need to append current generation info to history.
        # sol_history: (1, len, dims)
        # fit_history: (1, len, 1)
        # dist_history: (1, len, dims)
        
        # Current best solution or mean?
        # Usually we track the mean and its fitness?
        # Or the best in population?
        # EvoTF usually tracks "generations".
        # Let's track the current mean, current best fitness (or mean fitness), and maybe grad?
        # dist_features in EvoTF usually refers to distribution params (mean, std) or gradients.
        # Let's use current mean and global std?
        
        # Update buffers by rolling and replacing last element
        
        # Solution feature: current mean
        curr_sol = state.mean.reshape(1, 1, -1)
        new_sol_hist = jnp.concatenate([state.sol_history[:, 1:, :], curr_sol], axis=1)
        
        # Fitness feature: best fitness or mean fitness?
        # Let's use best fitness in this generation
        curr_fit = jnp.min(fitness).reshape(1, 1, 1)
        new_fit_hist = jnp.concatenate([state.fit_history[:, 1:, :], curr_fit], axis=1)
        
        # Dist feature: let's use the gradient we just computed? or the update vector?
        # Or just the global std?
        # Let's use the update vector (mean - old_mean)
        # Or just the gradient.
        curr_dist = grad.reshape(1, 1, -1)
        new_dist_hist = jnp.concatenate([state.dist_history[:, 1:, :], curr_dist], axis=1)
        
        # Update best fitness/solution tracking
        new_best_fit = jnp.min(fitness)
        best_idx = jnp.argmin(fitness)
        new_best_sol = population[best_idx]
        
        state = jax.lax.cond(
            new_best_fit < state.best_fitness,
            lambda st: st.replace(best_fitness=new_best_fit, best_solution=new_best_sol),
            lambda st: st,
            state
        )
        
        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter + 1),
            opt_state=opt_state,
            generation_counter=state.generation_counter + 1,
            noise_state=new_noise_state,
            sol_history=new_sol_hist,
            fit_history=new_fit_hist,
            dist_history=new_dist_hist,
        )

