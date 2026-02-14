import jax
import time
import datetime
import functools
from tqdm import tqdm
from flax import linen
from copy import deepcopy

from brax.envs.base import State as BraxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.types import Params

from legion.policy import Policy, register_policy
from legion.runtime import Runtime
from legion.logger import Logger

from .base import LearningRunner


class _BraxEnv:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self._validate_runtime()

    def _validate_runtime(self):
        if self.runtime.backend.name != "jax":
            raise ValueError(
                f"Can only use brax with 'jax' backend, not '{self.runtime.backend.name}'. Use a physics engine with compatible backend (used '{self.runtime.physics.name}')."
            )

    def reset(self, rng: jax.Array) -> BraxState:
        # Get initial state and observations
        state = self.runtime.reset(rng)
        obs = self.runtime.observe(state)

        # Initial reward and termination
        reward, done = self.runtime.backend.zeros(2)

        # Initial empty metrics
        metrics = {
            metric: self.runtime.backend.zeros(())
            for metric in self.runtime.task.metric_names_rewards
        }

        return BraxState(
            pipeline_state=state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def step(self, state: BraxState, action: jax.Array) -> BraxState:
        transition = self.runtime.step(state.pipeline_state, action)
        return state.replace(
            pipeline_state=transition.state,
            obs=transition.obs,
            reward=transition.reward,
            done=transition.done.astype("float32"),  # Brax expects float32s
            metrics=transition.metrics,
        )

    @property
    def observation_size(self) -> int:
        return self.runtime.backend.sum(
            self.runtime.backend.array([o.size for o in self.runtime.task.observations])
        )

    @property
    def action_size(self) -> int:
        return self.runtime.actuator.n_u

    @property
    def unwrapped(self) -> "_BraxEnv":
        return self


@register_policy("brax")
class BraxPolicy(Policy):
    def __init__(self, algo_cfg: dict, params: Params, obs_size: int, n_u: int):
        # Save configs (legion) and params (brax) for optional saving to file
        self.algo_cfg = algo_cfg
        self.obs_size = obs_size
        self.n_u = n_u
        self.params = params

        # NOTE: Currently we only support PPO
        if algo_cfg["type"] != "ppo":
            raise ValueError(
                f"BraxPolicy only supports 'ppo' for inference networks, not '{algo_cfg['type']}'"
            )

        # Build the PPO networks
        network_factory = _create_brax_network_factory(algo_cfg)
        extra_kwargs = (
            {"preprocess_observations_fn": running_statistics.normalize}
            if algo_cfg["normalize_observations"]
            else {}
        )  # Need to manually add this outside of training
        ppo_network = network_factory(obs_size, n_u, **extra_kwargs)

        # Create a jitted brax inference function as policy
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        self.inference_fn = jax.jit(make_inference_fn(params))

    def action(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        u, _ = self.inference_fn(obs, rng)
        return u

    def save_dict(self) -> dict:
        return {
            "algo_cfg": self.algo_cfg,
            "params": self.params,
            "obs_size": self.obs_size,
            "n_u": self.n_u,
        }

    @classmethod
    def load_from_dict(cls, data: dict) -> "BraxPolicy":
        return cls(data["algo_cfg"], data["params"], data["obs_size"], data["n_u"])


class BraxLearningRunner(LearningRunner):

    def learn_impl(self, runtime: Runtime, algo_cfg: dict, logger: Logger) -> Policy:
        # Create training function
        learn_fn = self._create_learn_fn(algo_cfg, logger)

        # Wrap the runtime
        env = _BraxEnv(runtime)

        # Perform learning
        start_time = time.perf_counter()
        make_inference_fn, params, metrics = learn_fn(environment=env)
        duration = time.perf_counter() - start_time
        print(f"Learning duration: {datetime.timedelta(seconds=duration)}")

        # Create policy
        return BraxPolicy(algo_cfg, params, env.observation_size, env.action_size)

    def _create_learn_fn(self, algo_cfg: dict, logger: Logger):
        # Edit a copy
        algo_cfg = deepcopy(algo_cfg)

        # Get algorithm type
        algo_type = algo_cfg["type"]

        # Create a progress bar
        pbar = tqdm(total=self.learning_iterations)

        # Create progress function called at each training iteration
        def progress_fn(num_steps: int, metrics: dict[str, float]):
            iteration = int(num_steps / self.batch_size)

            # Log metrics
            logger.log_metrics(iteration, metrics)

            # Update progress bar
            pbar.n = iteration
            pbar.refresh()

        if algo_type == "ppo":
            return functools.partial(
                ppo.train,
                num_timesteps=self.n_iterations,
                num_envs=self.n_envs,
                episode_length=1_000,  # FIXME: should be implemented in runtime
                action_repeat=1,
                learning_rate=algo_cfg["learning_rate"],
                entropy_cost=algo_cfg["entropy_coef"],
                discounting=algo_cfg["discount_factor"],
                unroll_length=self.rollout_length,
                batch_size=round(
                    self.batch_size / algo_cfg["n_mini_batches"] / self.rollout_length
                ),  # Internally brax multiplies batch size with number of minibatches and rollout length...
                num_minibatches=algo_cfg["n_mini_batches"],
                num_updates_per_batch=algo_cfg["n_learning_epochs"],
                normalize_observations=algo_cfg["normalize_observations"],
                reward_scaling=1.0,
                clipping_epsilon=algo_cfg["clip_range"],
                gae_lambda=algo_cfg["gae_lambda"],
                max_grad_norm=algo_cfg["max_grad_norm"],
                normalize_advantage=algo_cfg["normalize_advantage"],
                vf_loss_coefficient=algo_cfg["vf_loss_coefficient"],
                desired_kl=algo_cfg["desired_kl"],
                learning_rate_schedule={"adaptive_kl": "ADAPTIVE_KL", "none": "NONE"}[
                    algo_cfg["learning_rate_schedule"]
                ],
                network_factory=_create_brax_network_factory(algo_cfg),
                run_evals=False,
                progress_fn=progress_fn,
                log_training_metrics=True,
            )

        raise ValueError(f"Unsupported algorithm: {algo_type}")


def _create_brax_network_factory(algo_cfg: dict):
    algo_type = algo_cfg["type"]
    if algo_type == "ppo":
        return functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=algo_cfg["layers"],
            value_hidden_layer_sizes=algo_cfg["layers"],
            activation={
                "elu": linen.elu,
                "relu": linen.relu,
                "tanh": linen.tanh,
            }[algo_cfg["activation"]],
        )

    raise ValueError(f"Unsupported algorithm for brax network factory: {algo_type}")
