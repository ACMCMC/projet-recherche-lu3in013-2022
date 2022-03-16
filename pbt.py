import time
import numpy as np
import copy # used for multiprocessing

import gym
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

import salina
from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent, NRemoteAgent
from salina.agents.asynchronous import AsynchronousAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    # Create the workspace