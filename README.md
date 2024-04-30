# Practical RL - A Beginners Guide

## GymBasics
Gymnasium (once OpenAI Gym) is the go to for the creation of most basic reinforcement learning environments, containing both default environments and the ability to create custom environments with relative ease. Most frameworks are able to play nice with Gym environments, with many even requiring gym to function. Some of the environments can easily interface with Pygame if you want to interact with them directly. Actions can also be randomly sampled to see what an actor might do in a given environment.

This folder contains the basics of creating and running an environment, including an example custom gridworld environment and a template for creating your own environment. 

## RLlib
RLlib is the industry standard for large scale training and deployment of reinforcement learning models. This framework is extremely customizable, allowing almost any aspect of the reinforcement learning process to be modified according to your needs. Unfortunately, the framework has significant initial overheads and a massive learning curve. Furthermore, the framework is a hot mess, with many useless warnings often being thrown when beginning a training loop and esoteric documentation. There are occasional problems with functionality (such as rendering issues), and testing models trained by RLlib is non-trivial. However, the power of this framework is essentially unmatched by others, especially if you need to train across a cluster of CPU's or using multiple workers at once. 

This folder contains the basics necessary for training an agent using RLlib on both premade and custom environments. I am still working on adding good examples on custom callbacks (changes to the training process) and custom models.

## StableBaseline
StableBaseline 3 is a lightweight reinforcement learning framework. It works great with custom environments, with little to no overhead and extremely easy testing and deployment of models. However, it can be quite challenging to create a fully custom model, and it provides less overall control compared to RLlib. If you just want to train a model on your environment, then this is likely the framework for you. 

This folder contains the basics for training a policy on a custom environment. 
