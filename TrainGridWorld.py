from ray.rllib.algorithms.ppo import PPOConfig #Use Config objects rather than the algorithms directly
from ray.rllib.models import ModelCatalog
from ray import tune


from GymBasics.EnvironmentGridworld import CustomGridWorldEnv
from RLlib.CustomModel import CustomModel

#Register your custom model
ModelCatalog.register_custom_model("my_model", CustomModel)

#Register your custom environment
tune.register_env("Grid-v0", CustomGridWorldEnv)

#Code for creating a vanilla model to run with PPO
# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=1)
#     .resources(num_gpus=0)
#     .environment(env="Grid-v0")
#     .build()
# )

#Example of custom model initialization - Beware that this is currently dramatically worse than default models.
#I'm looking into what's causing this issue. 
algo = (
        PPOConfig()
        .environment(env="Grid-v0")
        .resources(num_gpus=1)
        .training(lr=0.01, grad_clip=30.0, 
                  model={
                        "custom_model": "my_model",
                        "custom_model_config": {
                            "obs_size": 12*12+2,
                            "action_size": 5,
                            "num_layers": 6,
                            },
                        }
        )
        .build()
        )
 


custom_checkpoint_dir = "./Checkpoints/GridworldModel"

print_guide = ["episodes_this_iter", "episode_reward_mean", "episode_reward_max", "episode_reward_min",
               "policy_reward_mean", "done",
               "num_env_steps_sampled_this_iter", ]

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