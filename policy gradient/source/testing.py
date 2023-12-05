
import torch
import numpy as np

def evaluate(agent, env, num_episodes=5, reinforce=True):
    stats = []
    with torch.no_grad():
        total_reward = 0.0
        for _ in range(num_episodes):
            state,_ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                if reinforce:
                    action, _ = agent.policy_net.get_action(state)
                else:
                    action, _, _ = agent.actor_critic(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            stats.append(episode_reward)
    print('\n\n------------------------------------------')
    print(f"Total reward: {total_reward}")
    print(f"Mean reward: {np.mean(stats)}")