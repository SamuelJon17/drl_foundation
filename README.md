# drl_foundation

This research project examines three Deep Reinforcement Learning (DRL) algorithms: Deep Q-Networks (DQN), REINFORCE, and Actor-Critic (A2C). DQN is good for high-dimensional spaces but struggles with continuous actions and overestimates Q-values. REINFORCE is effective in high-dimensional action spaces but faces issues with variance and sample efficiency. A2C, combining value and policy methods, offers stability but is complex. The project aims to develop these algorithms from scratch using PyTorch, following original methodologies, and evaluate their performance using OpenAI's Gymnasium -- CartPole & Ms PacMan.

## Environment Introdution
I use two environments to test the performance separately, such as [LunarLander](https://en.wikipedia.org/wiki/Lunar_Lander_(1979_video_game)), [Assault](https://en.wikipedia.org/wiki/Assault_(1983_video_game)), and [Mario](https://en.wikipedia.org/wiki/Super_Mario). You can click the hyperlinks to see the game rules. The following GIFs are my best results.

| CartPole - DQN  | CartPole - REINFORCE | CartPole - A2C |
| :--------: | :--------: | :--------: |
|<img src="static/DQN.gif"  height="200" /> |<img src="static/REINFORCE.gif"  height="200" /> |<img src="static/A2C.gif"  height="200" /> |


| Ms PacMan - Random  | Ms PacMan - DQN |
| :--------: | :--------: |
|<img src="static/pacman random.gif"  height="200" /> |<img src="static/pacman dqn.gif"  height="200" /> |
