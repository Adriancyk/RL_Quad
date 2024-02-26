import torch


def soft_update(target, source, tau):
    # Soft update model parameters.
    # θ_target = τ*θ_local + (1 - τ)*θ_target
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    # Hard update model parameters.
    # θ_target = θ_local
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)