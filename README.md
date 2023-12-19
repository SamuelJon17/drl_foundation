# Comparative Analysis & Implementation of Foundational Deep Reinforcement Learning Algorithms 

This research project examines three Deep Reinforcement Learning (DRL) algorithms: Deep Q-Networks (DQN), REINFORCE, and Actor-Critic (A2C). DQN is good for high-dimensional spaces but struggles with continuous actions and overestimates Q-values. REINFORCE is effective in high-dimensional action spaces but faces issues with variance and sample efficiency. A2C, combining value and policy methods, offers stability but is complex. The project aims to develop these algorithms from scratch using PyTorch, following original methodologies, and evaluate their performance using OpenAI's Gymnasium -- CartPole & Ms PacMan.

## Environment
I use two environments to test the performance separately, such as [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and [Ms PacMan](https://gymnasium.farama.org/environments/atari/ms_pacman/).

| CartPole | Ms PacMan|
| :--------: | :--------: |
|<img src="static/Random.gif" />| <img src="static/pacman random.gif" />|

## Usage
Training REINFORCE or A2C
* Navigate to *policy gradient/source/main_A2C.py or main_REINFORCE.py*
* Set TRAIN and TEST to True
* Feel free to change other hyperparameters such as number of episodes, episode limit, gamma, learning rate and the frequency logging results based on episodes.

Training DQN
* Navigate to *dqn/source/main_RAM.py or main.py*
* Main RAM uses a simple MLP and used for CartPole or other simple environments. Whereas Main uses a CNN followed by an MLP used for complex environments such as Ms PacMan
* Set TRAIN and TEST to True
* Feel free to change other hyperparameters such as number of episodes, episode limit, learning starts, policy update frequency, batch size, replay buffer size, gamma, learning rate and the frequency logging results based on episodes.

## Training Results
The training outcomes for the REINFORCE and hybrid Actor-Critic (A2C) methods on the CartPole environment are presented, averaged over 100 episodes. Notably, REINFORCE outperformed A2C by achieving an average reward near 450 in the final 500 episodes, indicating a faster convergence to an optimal policy. This could be attributed to differences in policy update mechanisms between the two methods. Visual analysis confirms that both agents effectively learned to balance the pole. It is hypothesized that extended training could elevate A2C’s performance to similar levels.

The correlation between the linear scheduler (black line) and the average reward over 1,000 episodes (green line) within the DQN training results suggests increased rewards correlating with a shift from exploration to exploitation. The red line indicates where training actually started, approximately 2.2K episodes (50K steps), with the target network updated 50 times over the course. Despite a longer training duration, the DQN agent’s policy, peaking at an average reward of just over 100 in the last 1,000 episodes, appears suboptimal, characterized by gradual directional movements without effective pole stabilization.
For MsPacman, the assessment is based solely on qualitative metrics. The DQN agent, compared to a random policy agent, showed an average reward increase from 20 to 200. However, across numerous episodes, the DQN agent consistently failed to complete the objective of collecting all pellets, indicating

| CartPole - Policy Gradient | CartPole - DQN |
| :--------: | :--------: |
|<img src="static/cartpole policy gradient.png" />| <img src="static/cartpole dqn.png" />|


Below are GIFs of each model's best results for both environments:

| CartPole - DQN  | CartPole - REINFORCE | CartPole - A2C |
| :--------: | :--------: | :--------: |
|<img src="static/DQN.gif"  height="200" /> |<img src="static/REINFORCE.gif"  height="200" /> |<img src="static/A2C.gif"  height="200" /> |


| Ms PacMan - DQN |
| :--------: |
|<img src="static/pacman dqn.gif" /> |
