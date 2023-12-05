import random
from collections import deque 
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        ****This class is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3

        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """ Value of the schedule at step t"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        ## deque is used as it automatically removes the oldest from the collection
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def __len__(self):
        return len(self.memory)
    
    def can_sample(self, batchSize):
        return batchSize <= self.__len__()
    
    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, requires_grad=False).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=False).to(DEVICE)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        assert self.can_sample(batch_size), "Batch size is larger than current memory size"
        
        # Randomly sampling 'batch_size' experiences
        batch = random.sample(self.memory, batch_size)

        # Unzipping the sampled batch
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE)
        rewards = torch.tensor(rewards).to(DEVICE)
        #next_states = torch.tensor(next_states, requires_grad=True).to(DEVICE)
        next_states = torch.stack(next_states).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.int).to(DEVICE)

        return states, actions, rewards, next_states, dones
