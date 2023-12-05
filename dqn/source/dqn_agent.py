from torch import optim
import torch.nn as nn
import torch
import numpy as np
from utils import LinearSchedule, ReplayBuffer
import os
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, model, state_dim=4, action_dim=8, learning_rate=0.00025, gamma=0.99, 
                 epsilon_start=1, epsilon_end=0.1):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = model(state_dim, self.action_dim).to(DEVICE)
        self.target_net = model(state_dim, self.action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=learning_rate, alpha=0.95, eps=0.01)
        self.loss =  nn.MSELoss()
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=300)
        self.exploration_schedule = LinearSchedule(1000000, epsilon_end, epsilon_start)

    def select_action(self, state, step):
        sample = np.random.rand()
        eps_threshold =  self.exploration_schedule.value(step)
        # Return the action with the highest Q-value predicted by the Q-network
        if sample > eps_threshold:
             with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return  self.q_net(state).argmax().item()
        # Otherwise, return random action       
        else:
            return np.random.choice(self.action_dim)

    def train(self, batch_size):
        ## Sample batchSize in ReplayMemory (random)
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        ## for i in batch:
            ## y_i = r_i if i.done
            ## else y_i = r_i + discount * argmax_a(QNet_target(phi_s_i+1))
            ## loss = (y_i - QNet_act(phi_s_i,action_i))**2
            ## gradient(loss)

        not_dones = 1-dones
        
        # Retrieves the Q-values predicted by the Q-network for the specific actions taken by the agent in the given states
        qnet_pred_currState = self.q_net(states).gather(1, actions.unsqueeze(1))

        # Retrieve max Q of the next state for each sample in batch, note different from above as the above is provided an action
        # for the corresponding state that the agent took where as this is identifying the max Q for all actions that state can take
        target_pred_nextState = self.target_net(next_states).detach().max(1)[0] 
        
        y_batch = (rewards + self.gamma*(not_dones*target_pred_nextState)).unsqueeze(1)
 
        self.optimizer.zero_grad()
        loss = self.loss(qnet_pred_currState, y_batch)
        loss.backward()
         # FROM PAPER (NEW)
        # loss = -1*(y_batch-qnet_pred_currState).clamp(-1, 1)
        # qnet_pred_currState.backward(loss)
        self.optimizer.step()

    def update_target_network(self):
        # Update the target Q-network weights
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def save(self, name):
        # Save model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        torch.save(self.target_net.state_dict(), path)
        return path

    def load(self, name):
        # Load model weights
        path = os.path.join('trained_models',name +'_weights.pth')
        self.q_net.load_state_dict(torch.load(path, map_location=DEVICE))
        # Sync the target network with the loaded main Q-network
        self.target_net.load_state_dict(self.q_net.state_dict())




