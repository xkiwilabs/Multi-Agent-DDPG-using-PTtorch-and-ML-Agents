
"""
Test DDPG Model for Unity ML-Agents Environments using PyTorch

The example uses a modified version of the Unity ML-Agents Tennis Example Environment.
In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, 
it receives a reward of -0.01. Thus, the goal of each agent is to keep 
the ball in play

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

###################################
# Import Required Packages
import torch
import time
import random
import numpy as np
from ddpg_agent import Agent
from unityagents import UnityEnvironment

"""
###################################
STEP 1: Set the Test Parameters
======
        num_episodes (int): number of test episodes
"""
num_episodes=2            


"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""
env = UnityEnvironment(file_name="Tennis.app")
# - **Mac**: "Tennis_Mac/Tennis.app"
# - **Windows** (x86): "Tennis_Windows_x86/Tennis.exe"
# - **Windows** (x86_64): "TennisWindows_x86_64/Tennis.exe"
# - **Linux** (x86): "Tennis_Linux/Tennis.x86"
# - **Linux** (x86_64): "Tennis_Linux/Tennis.x86_64"
# - **Linux** (x86, headless): "Tennis_Linux_NoVis/Tennis.x86"
# - **Linux** (x86_64, headless): "Tennis_Linux_NoVis/Tennis.x86_64"

"""
#######################################
STEP 3: Get The Unity Environment Brian
Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
the actions an agent or set of agents should take given a current set of environment (state) 
observations. The Reacher environment has a single Brian, thus, we just need to access the first brain 
available (i.e., the default brain). We then set the default brain as the brain that will be controlled.
"""
# Get the default brain 
brain_name = env.brain_names[0]

# Assign the default brain as the brain to be controlled
brain = env.brains[brain_name]


"""
#############################################
STEP 4: Determine the size of the Action and State Spaces and the Number of Agents

The observation space consists of 8 variables corresponding to the position and 
velocity of the ball and racket. Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) 
the net, and jumping.

The Tennis environment contains two agents in the environment. The task is a cooperative,
muli-agent task. 
"""

# Get number of agents in Environment
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
print('\nNumber of Agents: ', num_agents)

# Set the number of actions or action size
action_size = brain.vector_action_space_size

# Set the size of state observations or state size
states = env_info.vector_observations
state_size = states.shape[1]
print('\nSize of State: ', state_size)


"""
###################################
STEP 5: Initialize DDPG Agents from the Agent Class in dqn_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize two agents
We set the states size to 48 (24*2), so we can feed each agent boths agent's state observations.
"""
#Initialize Agent
agent_1 = Agent(state_size=48, action_size=action_size, num_agents=1, random_seed=0)
agent_2 = Agent(state_size=48, action_size=action_size, num_agents=1, random_seed=0)

# Load trained model weights for agent 1
agent_1.actor_local.load_state_dict(torch.load('ddpgActor1_Model.pth'))
agent_1.critic_local.load_state_dict(torch.load('ddpgCritic1_Model.pth'))

# Load trained model weights for agent 2
agent_2.actor_local.load_state_dict(torch.load('ddpgActor2_Model.pth'))
agent_2.critic_local.load_state_dict(torch.load('ddpgCritic2_Model.pth'))

"""
###################################
STEP 6: Play Banana for specified number of Episodes
"""
# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    # set train mode to false
    env_info = env.reset(train_mode=False)[brain_name]     

    # get initial state of the unity environment 
    states = env_info.vector_observations
    states = np.reshape(states, (1, 48))  # reshape so we can feed both agents states to each agent

    # reset the training agent for new episode
    agent_1.reset()
    agent_2.reset()

    # set the initial episode scores to zero for each unity agent.
    agent_scores = np.zeros(num_agents)

    # Run the episode loop;
    # At each loop step take an action as a function of the current state observations
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine actions for the unity agents from current sate
        actions_1 = agent_1.act(states, add_noise=False)
        actions_2 = agent_2.act(states, add_noise=False)

        # send the actions to the unity agents in the environment and receive resultant environment information
        actions = np.concatenate((actions_1, actions_2), axis=0) 
        actions = np.reshape(actions, (1, 4))
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations   # get the next states for each unity agent in the environment
        next_states = np.reshape(next_states, (1, 48))
        rewards = env_info.rewards                   # get the rewards for each unity agent in the environment
        dones = env_info.local_done                  # see if episode has finished for each unity agent in the environment

        # set new states to current states for determining next actions
        states = next_states

        # Update episode score for each unity agent
        agent_scores += rewards

        # If any unity agent indicates that the episode is done, 
        # then exit episode loop, to begin new episode
        if np.any(dones):
            break

    # Print current average score
    print('\nEpisode {}\tAgent1 Score: {:.2f}\tAgent2 Score: {:.2f}'.format(i_episode, agent_scores[0], agent_scores[1], end=""))


"""
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
env.close()

# END :) #############

