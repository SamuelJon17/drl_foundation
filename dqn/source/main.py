import gymnasium as gym
from dqn_agent import DQNAgent
from training import train_dqn
from testing import evaluate_dqn
from dqn_net import QNet

TRAIN=False
TEST=True

NUM_EPISODES = 2501 
EPISODE_LIMIT = 4000 # Episodic limit 
LOG_EVERY_N_EPISODES = 10 # Log
LEARNING_STARTS = 50000 # For Action-Value Q & Target, time steps
TARGER_UPDATE_FREQ = 10000 # For Target Action-Value Q, number of learning freq updates

LEARNING_FREQ = 4 # For Action-Value Q
BATCH_SIZE = 32 
REPLAY_BUFFER_SIZE = 1000000
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPS_START= 1
EPS_END = 0.01
FRAME_STACK_LEN = 4


if __name__ == '__main__':

    env = gym.make("MsPacmanNoFrameskip-v4")
    ##  First, to encode a single frame we take the maximum value for each pixel colour value over the frame being encoded and the previous frame. 
    ##  Second, we then extract the Y channel, also known as luminance, from the RGB frame and rescale it to 84 X 84
    ##  applies this preprocessing to the m most recent frames and stacks them to produce the input to the Q-function, in which m = 4
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=FRAME_STACK_LEN, 
                                       screen_size=84, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, FRAME_STACK_LEN)

    if TRAIN:
        agent = DQNAgent(model=QNet,
                        state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n, 
                        learning_rate=LEARNING_RATE, 
                        gamma=GAMMA,
                        epsilon_start=EPS_START, 
                        epsilon_end=EPS_END)
        
        train_dqn(agent, env, save_name='pacman_DQN',
                num_episodes=NUM_EPISODES, 
                episode_limit=EPISODE_LIMIT, 
                batch_size=BATCH_SIZE, 
                target_update_freq=TARGER_UPDATE_FREQ, 
                learning_starts=LEARNING_STARTS, 
                learning_freq=LEARNING_FREQ,
                LOG_EVERY_N_EPISODES = LOG_EVERY_N_EPISODES)
        env.close()
    else: 
        agent = DQNAgent(model=QNet,
                         state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n)
        agent.load('pacman_DQN')

    if TEST:
        env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")
        ##  First, to encode a single frame we take the maximum value for each pixel colour value over the frame being encoded and the previous frame. 
        ##  Second, we then extract the Y channel, also known as luminance, from the RGB frame and rescale it to 84 X 84
        ##  applies this preprocessing to the m most recent frames and stacks them to produce the input to the Q-function, in which m = 4
        env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=FRAME_STACK_LEN, 
                                            screen_size=84, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        env = gym.wrappers.FrameStack(env, FRAME_STACK_LEN)

        evaluate_dqn(agent, env, num_episodes=10)
        env.close()
        