import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_reinforce(agent, env, save_name = 'reinforce', 
                    num_episodes=5000, episode_limit=10000, LOG_EVERY_N_EPISODES=100):
    episodes_reward = []
    avg_episodes_reward= []
    episodes_step = []

    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        episode_reward = []
        episode_logProb = []
        steps = 0

        while not done and steps <= episode_limit:
            action, log_prob_action = agent.policy_net.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward.append(reward)
            episode_logProb.append(log_prob_action)
            state = next_state
            steps += 1

        agent.update_policy(episode_reward, episode_logProb)
        episodes_reward.append(sum(episode_reward))
        episodes_step.append(steps)
        
        if (episode % LOG_EVERY_N_EPISODES == 0):
            mean_reward = np.mean(episodes_reward[-LOG_EVERY_N_EPISODES:])
            avg_episodes_reward.append(mean_reward)
            best_reward = np.max(episodes_reward[-LOG_EVERY_N_EPISODES:])
            print('\n\n------------------------------------------')
            print(f"Episode {episode}/{num_episodes}")
            print(f"Mean reward since last {LOG_EVERY_N_EPISODES} episodes: {mean_reward}")
            print(f"Best reward over last {LOG_EVERY_N_EPISODES} episodes: {best_reward}")
            print(f"Episode steps: {steps}")


    print('Training Complete!')
    save_path = agent.save(save_name)
    print(f"Model saved at: {save_path}")
    plt.figure(figsize=(8,7))
    plt.plot(range(num_episodes), episodes_reward, c='blue', label='Reward', alpha=0.2)
    plt.plot(np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), avg_episodes_reward, c='blue', label=f'Average-{LOG_EVERY_N_EPISODES}-Reward')
    plt.title(f'REINFORCE Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures',save_name+'_figure.png'))
    plt.show()
    pd.DataFrame({'episode':range(num_episodes), 'reward':episodes_reward, 'step':episodes_step}).to_csv('episode_reinforce.csv')
    pd.DataFrame({'episode':np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), 'avg_reward':avg_episodes_reward}).to_csv('avg_episode_reward_reinforce.csv')


def train_a2c(agent, env, save_name = 'actor_critic', 
                       num_episodes=5000, episode_limit=10000, LOG_EVERY_N_EPISODES=100):
    
    episodes_reward = []
    avg_episodes_reward= []
    episodes_step = []
    total_steps = 0

    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        episode_reward = []
        episode_logProb = []
        episode_values = []
        episode_mask = []
        steps = 0
    
        while not done and steps <= episode_limit:
            action, log_prob_action, value = agent.actor_critic(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward.append(reward)
            episode_logProb.append(log_prob_action)
            episode_values.append(value)
            episode_mask.append(1-done)
            state = next_state
            steps += 1
            total_steps += 1

        _, _, next_value = agent.actor_critic(state)
        agent.update(episode_reward, episode_logProb, episode_values, episode_mask, next_value)
        episodes_reward.append(sum(episode_reward))
        episodes_step.append(steps)
        
        if (episode % LOG_EVERY_N_EPISODES == 0):
            mean_reward = np.mean(episodes_reward[-LOG_EVERY_N_EPISODES:])
            avg_episodes_reward.append(mean_reward)
            best_reward = np.max(episodes_reward[-LOG_EVERY_N_EPISODES:])
            print('\n\n------------------------------------------')
            print(f"Episode {episode}/{num_episodes}")
            print(f"Mean reward since last {LOG_EVERY_N_EPISODES} episodes: {mean_reward}")
            print(f"Best reward over last {LOG_EVERY_N_EPISODES} episodes: {best_reward}")
            print(f"Total steps: {total_steps}")


    print('Training Complete!')
    save_path = agent.save(save_name)
    print(f"Model saved at: {save_path}")
    plt.figure(figsize=(8,7))
    plt.plot(range(num_episodes), episodes_reward, c='blue', label='Reward', alpha=0.2)
    plt.plot(np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), avg_episodes_reward, c='blue', label=f'Average-{LOG_EVERY_N_EPISODES}-Reward')
    plt.title(f'A2C Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures',save_name+'_figure.png'))
    plt.show()
    pd.DataFrame({'episode':range(num_episodes), 'reward':episodes_reward, 'step':episodes_step}).to_csv('episode_a2c.csv')
    pd.DataFrame({'episode':np.arange(0, num_episodes, LOG_EVERY_N_EPISODES).tolist(), 'avg_reward':avg_episodes_reward}).to_csv('avg_episode_reward_a2c.csv')