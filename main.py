import torch
import numpy as np

from dynamics import QuadrotorEnv
import matplotlib.pyplot as plt
import os, sys

def train(agent, env, dynamics_model, args):