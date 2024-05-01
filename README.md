# Practical RL - A Beginners Guide
## Intro
The purpose of this repository is to provide a practical guide to getting started with your own reinforcement learning code. This will be most helpful if you have a decent understanding of RL already and are looking for a starting place on the practical side of things. The tutorials here are predominantly provided through the code and its comments, with the intent of people being able to copy and easily adapt it to their own situations with as little difficulty as possible. 

## Environments
### GymBasics
Gymnasium (once OpenAI Gym) is the go to for the creation of most basic reinforcement learning environments, containing both default environments and the ability to create custom environments with relative ease. Most frameworks are able to play nice with Gym environments, with many even requiring gym to function. Actions can also be randomly sampled to see what an actor might do in a given environment. Pygame is a simple way to create real time visualizations of gym environments. Many of the environments can easily interface with Pygame if you want to interact with them directly.

**Libraries:** Gymnasium, Pygame (optional, needed for gridworld)

This folder contains the basics of creating and running an environment, including an example custom gridworld environment and a template for creating your own environment. 

## Frameworks
### RLlib
RLlib is the "industry standard" for large scale training and deployment of reinforcement learning models. This framework is extremely customizable, allowing almost any aspect of the reinforcement learning process to be modified according to your needs (in theory). Unfortunately, this framework is notoriously hard to work with, including a massive learning curve.  

**Benefits:** Best for distributing over clusters, extremely customizable, includes implementations of most algorithms

**Problems:** Massive learning curve, significant overhead, terrible documentation, throws many non-fatal errors internally, non-trivial to test/deploy model.

**Use If:** You need to distribute over a cluster, need a massive level of customizability, 

**Libraries:** RLlib, Gymnasium, pygame (optional, needed for gridworld), pytorch or tensorflow (I use pytorch here)

This folder contains the basics necessary for training an agent using RLlib on both premade and custom environments. Furthermore, an example custom model (and how to use it for training) are provided. 

### StableBaseline
StableBaseline 3 is a lightweight reinforcement learning framework. It works great with custom environments, with little to no overhead and extremely easy testing and deployment of models. However, it is challenging to create a fully custom model (though some significant customization is possible), and it provides less overall control compared to RLlib. If you just want to train a model on your custom environment, then this is likely the framework for you. 

**Benefits:** Easy to use, lightweight, decent documentation, includes many of the more popular algorithms. Very easy to test/deploy

**Problems:** Not as much control as RLlib, harder to distribute over clusters

**Use If:** You want a simple yet powerful out of the box solution that is extremely easy to test/deploy

**Libraries:** StableBaseline, Gymnasium, pygame (optional, needed for gridworld)

This folder contains the basics for training a policy on a custom environment. 


