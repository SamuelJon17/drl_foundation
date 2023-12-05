import torch.optim as optim
import torch
import os
from nets import ActorCriticNet, PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class REINFORCE:
    def __init__(self, state_dim=4, action_dim=8, learning_rate=0.00025, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_net = PolicyNet(self.state_dim, self.action_dim, hidden_size=256).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def update_policy(self, episode_reward, episode_logProb):

        ## Doesn't need to be tracked, just used as a scalar, refer to notes and placement of nabla
        episode_reward = torch.tensor(episode_reward, dtype=torch.float32).to(DEVICE) 
        
        ## THIS LOSES TRACK OF INDIVIDUAL GRADIENTS DO NOT USE THIS APPROACH OF MAKING EPISODE_LOGPROB
        #episode_logProb = torch.tensor(episode_logProb, requires_grad=True, dtype=torch.float32).to(DEVICE)

        episode_logProb = [log_prob.unsqueeze(0) for log_prob in episode_logProb]
        episode_logProb = torch.cat(episode_logProb).flatten().to(DEVICE)
        
        G_t = []
        for t in range(len(episode_reward)):
            gamma = torch.FloatTensor([self.gamma ** (i + 1) for i in range(len(episode_reward[t:]))]).to(DEVICE)
            G_t.append(torch.dot(episode_reward[t:], gamma))

        G_t = torch.FloatTensor(G_t).to(DEVICE)
        G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-9) # normalize discounted reward, mean of 0 and a standard deviation of 1.
        
        self.optimizer.zero_grad()
        loss = -torch.dot(episode_logProb, G_t)
        loss.backward()
        self.optimizer.step()
        
    def save(self, name):
        # Save model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        torch.save(self.policy_net.state_dict(), path)
        return path

    def load(self, name):
        # Load model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))

class ActorCritic:
    def __init__(self, state_dim=4, action_dim=8, learning_rate=0.00025, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor_critic = ActorCriticNet(self.state_dim, self.action_dim, hidden_size=256).to(DEVICE)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

    def update(self, episode_reward, episode_logProb, episode_values, episode_mask, next_value):
        ## Doesn't need to be tracked, just used as a scalar, refer to notes and placement of nabla
        episode_reward = torch.tensor(episode_reward, dtype=torch.float32).to(DEVICE)  ## untracked
        # episode_logProb = [log_prob.unsqueeze(0) for log_prob in episode_logProb]
        episode_logProb = torch.cat(episode_logProb).to(DEVICE) ##tracked
        episode_mask = torch.tensor(episode_mask).to(DEVICE) ## untracked
        episode_values = torch.cat(episode_values).to(DEVICE) ## tracked

        episode_returns = []
        R = next_value
        for t in reversed(range(len(episode_reward))):
            R = episode_reward[t] + self.gamma * R * episode_mask[t]
            episode_returns.insert(0,R)
        episode_returns = torch.cat(episode_returns).detach() ##untracked

        episode_advantage = episode_returns - episode_values
        actor_loss = (-episode_logProb * episode_advantage).mean()
        critic_loss = (episode_advantage).pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        actor_critic_loss.backward()
        self.optimizer.step()
        
    def save(self, name):
        # Save model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        torch.save(self.actor_critic.state_dict(), path)
        return path

    def load(self, name):
        # Load model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        self.actor_critic.load_state_dict(torch.load(path, map_location=DEVICE))
