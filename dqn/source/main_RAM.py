import gymnasium as gym
from dqn_agent import DQNAgent
from training import train_dqn
from testing import evaluate_dqn
from dqn_net import DQN_RAM

TRAIN = False
TEST = True

NUM_EPISODES = 18001 # 
EPISODE_LIMIT = 500 # Limit episodic limit 
LOG_EVERY_N_EPISODES = 1000 # Log every 1000
LEARNING_STARTS = 50000 # For Action-Value Q & Target, time steps
TARGER_UPDATE_FREQ = 10000 # For Target Action-Value Q, number of learning freq updates

LEARNING_FREQ = 4 # For Action-Value Q
BATCH_SIZE = 32 
REPLAY_BUFFER_SIZE = 1000000
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPS_START= 1
EPS_END = 0.1


if __name__ == '__main__':

    env = gym.make("CartPole-v1")

    if TRAIN:
        agent = DQNAgent(model=DQN_RAM,
                        state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n, 
                        learning_rate=LEARNING_RATE, 
                        gamma=GAMMA,
                        epsilon_start=EPS_START, 
                        epsilon_end=EPS_END)
                
        train_dqn(agent, env, save_name='cartpole_DQN',
                num_episodes=NUM_EPISODES, 
                episode_limit=EPISODE_LIMIT, 
                batch_size=BATCH_SIZE, 
                target_update_freq=TARGER_UPDATE_FREQ, 
                learning_starts=LEARNING_STARTS, 
                learning_freq=LEARNING_FREQ,
                LOG_EVERY_N_EPISODES = LOG_EVERY_N_EPISODES)
        env.close()
    
    else:
        agent = DQNAgent(model=DQN_RAM,
                         state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n)
        agent.load('cartpole_DQN')

    if TEST:
        env = gym.make("CartPole-v1", render_mode="human")
        evaluate_dqn(agent, env, num_episodes=5)
        env.close()
    


