import gymnasium as gym
from agents import REINFORCE
from training import train_reinforce
from testing import evaluate

TRAIN = False
TEST = True

NUM_EPISODES = 2501
EPISODE_LIMIT = 500
GAMMA = 0.99
LEARNING_RATE = 0.00025
LOG_EVERY_N_EPISODES=100


if __name__ == '__main__':

    env = gym.make("CartPole-v1")

    if TRAIN:
        agent = REINFORCE(state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n, 
                        learning_rate=LEARNING_RATE, 
                        gamma=GAMMA)
        
        train_reinforce(agent, env, save_name='cartpole_REINFORCE',
                num_episodes=NUM_EPISODES, 
                episode_limit=EPISODE_LIMIT,
                LOG_EVERY_N_EPISODES=LOG_EVERY_N_EPISODES)
        env.close()
    
    else:
        agent = REINFORCE(state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n)
        agent.load('cartpole_REINFORCE')

    if TEST:
        env = gym.make("CartPole-v1", render_mode="human")
        evaluate(agent, env, num_episodes=10, reinforce=True)
        env.close()
    


