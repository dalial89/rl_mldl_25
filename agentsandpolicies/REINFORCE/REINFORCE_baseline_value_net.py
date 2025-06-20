import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Monte-Carlo return  G_t = Σ_{k=t+1}^{T} γ^{k-t-1} R_k      ← eq. (first line)
# ---------------------------------------------------------------------------
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# ---------------------------------------------------------------------------
#  π(a|s,θ)  and  v̂(s,w)
# ---------------------------------------------------------------------------
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
        self.fc1_critic  = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic  = torch.nn.Linear(self.hidden,  self.hidden)
        self.fc3_value   = torch.nn.Linear(self.hidden,  1) #scalar value


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        x_v = self.tanh(self.fc1_critic(x))
        x_v = self.tanh(self.fc2_critic(x_v))
        value = self.fc3_value(x_v).squeeze(-1)

        
        return normal_dist, value


class Agent:
    def __init__(self, policy, device='cpu', max_grad_norm=None):
        self.train_device = device
        self.policy = policy.to(device)

        actor_params  = [p for n,p in self.policy.named_parameters()
                         if n.startswith(("fc1_actor","fc2_actor","fc3_actor","sigma"))]
        critic_params = [p for n,p in self.policy.named_parameters()
                         if n.startswith(("fc1_critic","fc2_critic","fc3_value"))]

        self.opt_theta = torch.optim.Adam(actor_params , lr=3e-4)  # α^θ
        self.opt_w     = torch.optim.Adam(critic_params, lr=1e-3)  # α^w

        self.max_grad_norm = max_grad_norm
        self.gamma = 0.99

        self.states, self.action_log_probs, self.rewards = [], [], []


    def update_policy(self):
        states   = torch.stack(self.states).to(self.train_device)
        logps    = torch.stack(self.action_log_probs).to(self.train_device)
        rewards  = torch.stack(self.rewards).to(self.train_device).squeeze(-1)

        returns = discount_rewards(rewards, self.gamma) # G_t

        # δ_t = G_t − v̂(S_t,w)
        with torch.no_grad():
            _, state_values = self.policy(states)     

        delta     = returns - state_values 

        #Critic:  w ← w + α^w δ ∇_w v̂(S_t,w)
        _, v_pred = self.policy(states)                                 
        value_loss = 0.5 * (returns - v_pred).pow(2).mean()

        self.opt_w.zero_grad()
        value_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.opt_w.step()

        #Actor:  θ ← θ + α^θ γ^t δ ∇_θ logπ(A_t|S_t,θ)
        T = returns.size(0)
        discounts = self.gamma ** torch.arange(T, dtype=returns.dtype,
                                               device=self.train_device)
        actor_loss = -(discounts * delta.detach() * logps).mean()

        self.opt_theta.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.opt_theta.step()

        self.states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()


        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, action_log_prob, reward):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
