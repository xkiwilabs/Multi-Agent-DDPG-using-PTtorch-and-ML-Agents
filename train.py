
"""
DDPG (Actor-Critic) RL Example for Unity ML-Agents Environments using PyTorch
Includes examples of the following DDPG training algorithms:

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
import random
import numpy as np
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment

"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        episode_scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
        solved_score (float): the average score required for the environment to be considered solved
    """
num_episodes=10000
episode_scores = []
scores_average_window = 100      
solved_score = 1.0     #(sloved score is 0.5; I set it higher here to achive more robust end-sate performance)

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
STEP 5: Create DDPG Agents from the Agent Class in ddpg_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize two agents
We set the states size to 48 (24*2), so we can feed each agent boths agent's state observations.
"""
agent_1 = Agent(state_size=48, action_size=action_size, num_agents=1, random_seed=0)
agent_2 = Agent(state_size=48, action_size=action_size, num_agents=1, random_seed=0)
"""

###################################
STEP 6: Run the DDPG Training Sequence
The DDPG Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Perform an action, a(t), in the environment given s(t)
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Update agent memory and learn from experience (i.e, agent.step)
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).

Below we also exit the training process early if the environment is solved. 
That is, if the average score for the previous 100 episodes is greater than solved_score.
"""

# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    env_info = env.reset(train_mode=True)[brain_name]     

    # get initial state of the unity environment 
    states = env_info.vector_observations
    states = np.reshape(states, (1, 48)) # reshape so we can feed both agents states to each agent
 
	# reset each agent for a new episode
    agent_1.reset()
    agent_2.reset()

    # set the initial episode score to zero.
    agent_scores = np.zeros(num_agents)

    # Run the episode training loop;
    # At each loop step take an action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agents Actor and Critic networks
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine actions for the unity agents from current sate, using noise for exploration
        actions_1 = agent_1.act(states, add_noise=True)
        actions_2 = agent_2.act(states, add_noise=True)

        # send the actions to the unity agents in the environment and receive resultant environment information
        actions = np.concatenate((actions_1, actions_2), axis=0) 
        actions = np.reshape(actions, (1, 4))
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations   # get the next states for each unity agent in the environment
        next_states = np.reshape(next_states, (1, 48))
        rewards = env_info.rewards                   # get the rewards for each unity agent in the environment
        dones = env_info.local_done                  # see if episode has finished for each unity agent in the environment

        #Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
        agent_1.step(states, actions_1, rewards[0], next_states, dones[0])
        agent_2.step(states, actions_2, rewards[1], next_states, dones[1])

        # set new states to current states for determining next actions
        states = next_states
        #print(states)
        # Update episode score for each unity agent
        agent_scores += rewards

        # If any unity agent indicates that the episode is done, 
        # then exit episode loop, to begin new episode
        if np.any(dones):
            break

    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes 
    # Mean score is calculated over current episodes until i_episode > 100
    episode_scores.append(np.max(agent_scores))
    average_score = np.mean(episode_scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    #Print current and average score
    print('\nEpisode {}\tMax Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, episode_scores[i_episode-1], average_score), end="")
    
    # Save trained  Actor and Critic network weights for agent 1
    an_filename = "ddpgActor1_Model.pth"
    torch.save(agent_1.actor_local.state_dict(), an_filename)
    cn_filename = "ddpgCritic1_Model.pth"
    torch.save(agent_1.critic_local.state_dict(), cn_filename)

    # Save trained  Actor and Critic network weights for agent 2
    an_filename = "ddpgActor2_Model.pth"
    torch.save(agent_2.actor_local.state_dict(), an_filename)
    cn_filename = "ddpgCritic2_Model.pth"
    torch.save(agent_2.critic_local.state_dict(), cn_filename)

    # Check to see if the task is solved (i.e,. avearge_score > solved_score over 100 episodes). 
    # If yes, save the network weights and scores and end training.
    if i_episode > 100 and average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, average_score))

        # Save the recorded Scores data
        scores_filename = "ddpgAgent_Scores.csv"
        np.savetxt(scores_filename, episode_scores, delimiter=",")
        break


"""
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
env.close()

# END :) #############

