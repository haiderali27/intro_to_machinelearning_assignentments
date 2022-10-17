from posix import environ
import gym
import time
import numpy as np
import matplotlib.pyplot as plt


def eval_policy(qtable_, num_of_episodes_, max_steps_, env):
    rewards = []

    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
        
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward





def qLearningTaskDeterministic(episode, iters, steps, gamma, is_slip):
    env = gym.make("FrozenLake-v1",is_slippery=is_slip)
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Qtable = np.random.rand(n_observations,n_actions)
    rewards_per_episode = list()
    reward_best = -1000
    avg_rewards=[]
    num_of_episode=[]
    current_episode=1
    for e in range(episode):
        current_state = env.reset()
        done = False
    
        reward_tot = 0
    
        for i in range(iters): 
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            Qtable[current_state, action] = reward + gamma*np.max(Qtable[next_state,:])
            reward_tot = reward_tot + reward
            if done:
                break
            current_state = next_state
        

        if current_episode % steps == 0:
            x=0
            qtable_best=Qtable
            avg_rewards.append(eval_policy(qtable_best, 10, 100, env))
            num_of_episode.append(current_episode)

        rewards_per_episode.append(reward_tot)
        current_episode+=1
    
    return rewards_per_episode, avg_rewards, num_of_episode



def qLearningTaskNonDeterministic(episode, iters, steps, alpha, gamma, is_slip):
    env = gym.make("FrozenLake-v1",is_slippery=is_slip)
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Qtable = np.random.rand(n_observations,n_actions)
    rewards_per_episode = list()
    reward_best = -1000
    avg_rewards=[]
    num_of_episode=[]
    current_episode=1
    for e in range(episode):
        current_state = env.reset()
        done = False
    
        reward_tot = 0
    
        for i in range(iters): 
            count=-1
            
            action = env.action_space.sample()
    
            next_state, reward, done, _ = env.step(action)

            Qtable[current_state, action] = Qtable[current_state, action] + alpha*(reward + gamma*np.max(Qtable[next_state,:])-Qtable[current_state, action])
            reward_tot += reward
            if done:
                break
            current_state = next_state

    
        if current_episode % steps == 0:
            qtable_best=Qtable
            avg_rewards.append(eval_policy(qtable_best,10,1000, env))
            num_of_episode.append(current_episode)
        

        rewards_per_episode.append(reward_tot)
        current_episode+=1
   
    return rewards_per_episode,avg_rewards,num_of_episode

episodes=1000
iterations=100
steps=10
gamma=0.9
alpha=0.5
fig, axs= plt.subplots(3, 1)

for i in range(10):
    deterministicResult, avg_rewards, num_of_episode= qLearningTaskDeterministic(episodes,iterations, steps, gamma, False)
    j=i+1;
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Avg Rewards')
    axs[0].plot(avg_rewards, label = f'Plot {j}')
    #axs[0].plot(num_of_episode, avg_rewards, label = f'Plot {j}')
    axs[0].set_title(f'Slipery False with Deterministic Rule, with Episode:{episodes}, Iterations:{iterations}, gamma:{gamma}')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))


for i in range(10):
    deterministicResult, avg_rewards, num_of_episode= qLearningTaskDeterministic(episodes,iterations, steps, gamma, True)
    j=i+1;
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Avg Rewards')
    axs[1].plot(avg_rewards, label = f'Plot {j}')
    #axs[1].plot(num_of_episode, avg_rewards, label = f'Plot {j}')
    axs[1].set_title(f'Slipery True with Deterministic Rule, with Episode:{episodes}, Iterations:{iterations}, gamma:{gamma}')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))




for i in range(10):
    nonDeterministicResult, avg_rewards, num_of_episode = qLearningTaskNonDeterministic(episodes,iterations, steps, alpha, gamma, True)
    j=i+1;
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Avg Rewards')
    axs[2].plot(avg_rewards, label = f'Plot {j}')
    #axs[2].plot(num_of_episode, avg_rewards, label = f'Plot {j}')

    axs[2].set_title(f'Slipery True with Non Deterministic Rule, with Episode:{episodes}, Iterations:{iterations}, alpha:{alpha}, gamma:{gamma}')
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()

