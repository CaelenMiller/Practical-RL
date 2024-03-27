from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray import tune

from GymBasics.EnvironmentGridworld import CustomGridWorldEnv

tune.register_env("Grid-v0", CustomGridWorldEnv)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="Grid-v0")
    .build()
)

custom_checkpoint_dir = "./Checkpoints/GridworldModel"

print_guide = ["episodes_this_iter", "episode_reward_mean", "episode_reward_max", "episode_reward_min",
               "policy_reward_mean", "done",
               "num_env_steps_sampled_this_iter", ]

mean_rewards = []
print("BEGINNING TRAINING")
for i in range(1, 21):
    result = algo.train()
    print(f"Iteration: {i}")
    for key in print_guide:
        print(f'{key}: {result[key]}')
    if i % 20 == 0:
        checkpoint_dir = algo.save(custom_checkpoint_dir).checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
    print("\n")