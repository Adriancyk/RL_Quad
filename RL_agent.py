import torch
import torch.nn.functional as F
import numpy as np
from model import GaussianPolicy, QNetwork
from torch.optim import Adam
from utils import soft_update, hard_update


class SAC(object):

    def __init__(self, num_inputs, action_space, env, args):
        self.env = env
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)


        self.actor = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi = self.actor.sample(state_batch) 

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()
    
    def save_model(self, path):
        torch.save(self.critic.state_dict(), path + '_critic')
        torch.save(self.critic_optim.state_dict(), path + '_critic_optim')
        torch.save(self.actor.state_dict(), path + '_actor')
        torch.save(self.actor_optim.state_dict(), path + '_actor_optim')

    def load_model(self, path):
        self.critic.load_state_dict(torch.load(path + '_critic'))
        self.critic_optim.load_state_dict(torch.load(path + '_critic_optim'))
        self.critic_target = QNetwork(self.num_inputs, self.num_actions, self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor.load_state_dict(torch.load(path + '_actor'))
        self.actor_optim.load_state_dict(torch.load(path + '_actor_optim'))


