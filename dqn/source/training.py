import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dqn(agent, env, save_name = 'save_name', num_episodes=15000, episode_limit=10000, 
              batch_size=32, target_update_freq=10000, learning_starts=5000, learning_freq=4,
              LOG_EVERY_N_EPISODES=100):
## Training
## Reset environment
## For episode 1 to M
    ## For t=1 to Steps
        ## action = EpsilonGreedy(Q)
        ## state, reward, done, ... =  env.step(action)
        ## ReplayMemory.append((state,action,reward))
        ## UpdateMiniBatch()   

    episodes_reward = []
    avg_episodes_reward= []
    episodes_step = []
    episodes_eps = []
    num_learning = 0
    total_steps = 0
    target_update=0
    learning_starts_ep = -1
    
    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        episode_reward = []
        steps = 0
        
        while not done and steps < episode_limit:
            
            # Until steps > learning_starts
            if total_steps < learning_starts:
                action = np.random.choice(agent.action_dim)
            else:
                action = agent.select_action(state, total_steps)
            
            next_state, reward, done, _, _ = env.step(action)
            
            # CLIP reward between -1 and 1
            reward = max(-1.0, min(reward, 1.0))

            # Store experience
            agent.memory.push(state ,action, reward, next_state, done)
            state = next_state

            # Update episode_reward, etc.
            episode_reward.append(reward)
            total_steps+=1
            steps += 1
            
            # Train agent
            if (learning_starts < total_steps and 
                total_steps % learning_freq == 0 and
                agent.memory.can_sample(batch_size)):
                agent.train(batch_size)
                num_learning +=1
                if learning_starts_ep == -1:
                    print(f"TRAINING STARTED AT EP {episode}")
                    learning_starts_ep = episode

            # Update target network periodically
            if num_learning % target_update_freq == 0:
                agent.update_target_network()
                target_update+=1
        
        episodes_reward.append(sum(episode_reward))
        episodes_eps.append(agent.exploration_schedule.value(total_steps))
        episodes_step.append(steps)
        if (episode % LOG_EVERY_N_EPISODES == 0):
            
            mean_reward = np.mean(episodes_reward[-LOG_EVERY_N_EPISODES:])
            avg_episodes_reward.append(mean_reward)
            best_reward = np.max(episodes_reward[-LOG_EVERY_N_EPISODES:])
            print('\n\n------------------------------------------')
            print(f"Episode {episode}/{num_episodes}")
            print(f"Mean reward since last {LOG_EVERY_N_EPISODES} episodes: {mean_reward}")
            print(f"Best reward over last {LOG_EVERY_N_EPISODES} episodes: {best_reward}")
            print(total_steps)
            print(agent.exploration_schedule.value(total_steps)) 
            save_path = agent.save(save_name)
            print(f"Model saved at: {save_path}")
            
    print("Training completed!")
    print(f"Learning Started: {learning_starts_ep}")
    print(f"Total Number of Learnings: {num_learning}")
    print(f"Total number of target q-net updates: {target_update}")
    
    
    plt.figure(figsize=(8,7))
    plt.plot(range(num_episodes), episodes_reward, c='blue', label='Reward', alpha=0.2)
    plt.plot(np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), avg_episodes_reward, c='blue', label=f'Average-{LOG_EVERY_N_EPISODES}-Reward')
    plt.axvline(x=learning_starts_ep, color='r', label='Learning Starts')
    plt.title(f'DQN Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures',save_name+'_figure.png'))
    plt.show()
    save_path = agent.save(save_name)
    print(f"Model saved at: {save_path}")
    pd.DataFrame({'episode':range(num_episodes), 'reward':episodes_reward, 'step':episodes_step, 'eps':episodes_eps}).to_csv('episode_dqn.csv')
    pd.DataFrame({'episode':np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), 'avg_reward':avg_episodes_reward}).to_csv('avg_episode_reward_dqn.csv')

