import torch
import numpy as np

from dynamics import QuadrotorEnv
from RL_agent import SAC
import matplotlib.pyplot as plt
import os, sys

def train(agent, env, dynamics_model, args):










env = QuadrotorEnv()
agent = SAC(env.observation_space.shape[0], env.action_space, args)