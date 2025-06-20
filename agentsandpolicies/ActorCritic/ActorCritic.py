import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        #Returns both the action distribution and the state value.
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)  # shape (action_space,)
        # `Normal broadcasts sigma over the batch dimension
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)

        
        return normal_dist, state_value   # (distribution, V(s))


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99

        #Roll out buffers
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []  # 1 if episode finished after transition else 0


    def update_policy(self):
        """
        Using the one‑step TD actor‑critic update from the book:
            δ_t  = R_t + γ V(s_{t+1}) − V(s_t)
            θ    ← θ + α_θ  I_t  δ_t  ∇_θ log π_θ(a_t|s_t)
            w    ← w + α_w        δ_t  ∇_w V_w(s_t)
        """
        #from buffers I transform them in tensors
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        #clear buffers for the next episode
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # TASK 3:
        # -------- Critic forward pass -------------------
        _, state_values = self.policy(states)                  # V(s_t)
        with torch.no_grad():
            _, next_state_values = self.policy(next_states)    # V(s_{t+1})
            # If the episode finished at step t, bootstrap should stop there
            next_state_values = next_state_values * (1.0 - done)
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        returns = rewards + self.gamma * next_state_values        # R_t + γ V(s_{t+1})
        advantages = returns - state_values                       # δ_t
        #   - compute actor loss and critic loss
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values, returns)
        total_loss = actor_loss + critic_loss
        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_loss": total_loss.item(),
        }

    def get_action(self, state, evaluation=False):
        #given a state (np array) → sample/return action (+ log‑prob)
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)