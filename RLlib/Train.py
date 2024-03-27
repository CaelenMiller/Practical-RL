from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt

def plot(name, x_lab, y_lab, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(name)
    plt.show()

# Initialization of the algorithm. Workers are instances of your environment, lots of other variables you can set here
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env='CartPole-v1')
    .build()
)

# I would recommend using a custom directory, otherwise the model will be difficult to track down
custom_checkpoint_dir = "./Checkpoints/Model"

# A few stats that are useful for determining how the training is going. use result.keys() to see other options
print_guide = ["episodes_this_iter", "episode_reward_mean", "episode_reward_max", "episode_reward_min",
               "policy_reward_mean", "done",
               "num_env_steps_sampled_this_iter", ]

#nifty if you want to keep track of how training is going
mean_rewards = []

print("BEGINNING TRAINING")
for i in range(1, 21):
    result = algo.train()
    print(f"Iteration: {i}")
    for key in print_guide:
        print(f'{key}: {result[key]}')

    mean_rewards.append(result["episode_reward_mean"])

    if i % 20 == 0:
        checkpoint_dir = algo.save(custom_checkpoint_dir).checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
    print("\n")

plot("Mean Rewards", "Mean Rewards", "Training Episode", mean_rewards)


