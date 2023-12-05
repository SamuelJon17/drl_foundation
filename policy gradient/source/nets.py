import torch.nn as nn
import torch.nn.functional as F
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, state=4, num_actions=18, hidden_size=256):
        """
        Simple Policy Network for Low dimensional state spaces such as Cart Pole
            state: number of features in the state/observation space.
            num_actions: number of actions to output, one-to-one correspondence to action in game.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) ## column-wise
    
    def get_action(self, state):
        # Returns an action sampled from the probablity distribution returned from PolicyNet
        # Returns the log probablity of the action sampled -- used to update the parameters
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        prob_actions  = self.forward(state) ## returns a vector of probablities for each action
        categorical = torch.distributions.Categorical(prob_actions)
        stochastic_action = categorical.sample()
        log_prob_action = categorical.log_prob(stochastic_action)
        # stochastic_action = torch.multinomial(prob_actions, num_samples=1, replacement=True).item()
        # log_prob_action = torch.log(prob_actions.squeeze(0)[stochastic_action])
        # return stochastic_action, log_prob_action
        return stochastic_action.item(), log_prob_action


class ActorCriticNet(nn.Module):
    def __init__(self, state=4, num_actions=18, hidden_size=256):
        super(ActorCriticNet, self).__init__()
        
        self.actor_fc1 = nn.Linear(state, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, num_actions)

        self.critic_fc1 = nn.Linear(state, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        state = torch.Tensor(x).unsqueeze(0).to(DEVICE)
        
        prob_actions = F.relu(self.actor_fc1(state))
        prob_actions = F.softmax(self.actor_fc2(prob_actions), dim=1)

        categorical = torch.distributions.Categorical(prob_actions)
        stochastic_action = categorical.sample()
        log_prob_action = categorical.log_prob(stochastic_action)
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        return stochastic_action.item(), log_prob_action, value
