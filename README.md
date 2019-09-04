<img src="https://camo.githubusercontent.com/7ad5cdff66f7229c4e9822882b3c8e57960dca4e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4a756e652f35623165613737385f726561636865722f726561636865722e676966">   

### Deep Reinforcement Learning Reacher Continuous Control   
### Introduction
In this project, I build a reinforcement learning (RL) agent that controls a robotic arm within Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The goal is to get 20 different robotic arms to maintain contact with the green spheres.

A reward of +0.1 is provided for each timestep that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

In order to solve the environment, our agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
2. Version 1 (1 agent):
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
2. Version 2 (20 agents)
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.
### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0

I have also created a requirement.txt. You may do a pip install requirement.txt to install all the required packages.

### Execution 
To run my code, I have copied my entire code to main.py. So you need to execute the command python main.py

## Summary of Environment
- Set-up: Double-jointed arm which can move to target locations.
- Goal: Each agent must move its hand to the goal location, and keep it there.
- Agents: The environment contains 20 agents linked to a single Brain.
- Agent Reward Function (independent):
  - +0.1 for each timestep agent's hand is in goal location.
- Brains: One Brain with the following observation/action space.
  - Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
  - Visual Observations: None.
- Reset Parameters: Two, corresponding to goal size, and goal movement speed.
- Benchmark Mean Reward: 30
- 
## Approach
Here are the high-level steps taken in building an agent that solves this environment.

1. Evaluate the state and action space.
1. Establish performance baseline using a random action policy.
1. Select an appropriate algorithm and begin implementing it.
1. Run experiments, make revisions, and retrain the agent until the performance threshold is reached.
## Solution:
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Reacher environment. Second, we need to determine how many "brains" we want controlling the actions of our agents.

There are 2 main differences in the Reacher Environment:
1> Continuous Actions
2> Multi Agent

The value based methods (DQN etc) is not suitable for continuous action space.
Policy based menthods are well suited for this purpose.

But Policy based menthods uses Monte Carlo menthods. THis increases variance and we had to wait for the entire episode to complete to do the training. So I wanted to use a Policy based menthod where we will Temporal difference so that we can training for all the timeframe and the convergence is fast.
The algorithm I choose is Deep Deterministic Policy Gradient (DDPG). It is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy.Using a value-based approach, the agent (critic) learns how to estimate the Q value (reward). So we moved ahead of Monte Carlo which leads to too much ocillation.

The robot arm has different magnitude and direction as actions. So we cannot used random sampling, else the direction action would have a mean of 0, leading to not much learning.

So we used Ornstein-Uhlenbeck process. This process add a small amount of random noise to the action at each timestep.This noise is correlated to the previous noise, and hence tends to be in the same direction for longer durations which leads to higher score.

This noise is ignored during testing.
### The hyperparameters: 
- The file with the hyperparameters configuration is mentioned in the  <b>.ipnb</b>.  - If you want you can change the model configuration , change the Actor and Critic Class in the  <b>.ipnb</b> file. - The actual configuration of the hyperparameters is:    - Learning Rate: 1e-4 (in both DNN)   - Batch Size: 128   - Replay Buffer: 1e5   - Gamma: 0.99   - Tau: 1e-3   - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)  - For the neural models:       - Actor         - Hidden: (input, 256)  - ReLU     - Hidden: (256, 128)    - ReLU     - Output: (128, 4)      - TanH. The action output is clipped between -1 and 1.   - Critic     - Hidden: (input, 256)              - ReLU     - Hidden: (256 + action_size, 128)  - ReLU     - Output: (128, 1)                  - Linear

### Score vs Episode while training 1 agent
<img src="https://github.com/kaustav1987/Continuous-Control-using-DDPG/blob/master/Single%20Agent%20Score%20Plot.png"> 
It took me 263 episodes to get the desired score for Version1 (1 agent)

### Score vs Episode while training 20 agents
<img src="https://github.com/kaustav1987/Continuous-Control-using-DDPG/blob/master/20%20Agent%20Score%20Plot.png"> 
It took me 102 episodes to get the desired score for Version2 (20 agents)

I stil want to check this task with the A2C , D4PG algorithm and discover when and where each of the algorithms (DDPG vs. D4PG) have the best performance. I also want to check if using Advantage Critic benefits this task. I want to explore I want to try with experienced replay. We may learn more from rare but important events in that case.I think we may also try N-Step boostrapping instead 1 step for bias-variance tradeoff. I think the reward calculation using GAE may also benefit.


