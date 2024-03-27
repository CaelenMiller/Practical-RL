from ray.rllib.algorithms.ppo import PPOConfig

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env='CartPole-v1')
    .build()
)

for i in range(1, 21):
    result = algo.train()
    if i % 20 == 0:
        checkpoint_dir = algo.save().checkpoint.path
