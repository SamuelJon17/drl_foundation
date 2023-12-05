import torch
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dqn(agent, env, num_episodes=10):
    """
    Evaluate the performance of the DQNAgent without exploration.
    
    Parameters:
    - agent: the DQNAgent to be evaluated.
    - env: the environment where the agent acts.
    - num_episodes: number of episodes to evaluate the agent.
    
    Returns:
    - Average reward over num_episodes.
    """
    stats = []
    with torch.no_grad():
        total_reward = 0.0
        for _ in range(num_episodes):
            state,_ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = agent.q_net(state)
                action = q_values.argmax().item()

                # Step in the environment
                next_state, reward, done, _, _ = env.step(action)
                
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            stats.append(episode_reward)
    print('\n\n------------------------------------------')
    print(f"Total reward: {total_reward}")
    print(f"Mean reward: {np.mean(stats)}")
    print(f"Best reward: {np.max(stats)}")



