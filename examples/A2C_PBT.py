import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra

import random

import gym
# The TimeLimit wrapper is useful to specify a max number of steps for an RL episode
from gym.wrappers import TimeLimit
import salina

# Following Ludovic Denoyer's advice, we use Agent rather than TAgent
# `TAgent` is used as a convention 
# to represent agents that use a time index in their `__call__` function (not mandatory)
from salina import Agent, get_arguments, get_class, instantiate_class

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent (e.g a TAgent) over multiple timesteps in the workspace, 
# or until a given condition is reached
from salina.agents import Agents, RemoteAgent, TemporalAgent

# GymAgent (resp. AutoResetGymAgent) are agents able to execute a batch of gym environments
# without (resp. with) auto-resetting. These agents produce multiple variables in the workspace: 
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’, 
# ... When called at timestep t=0, then the environments are automatically reset. 
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from salina.agents.gyma import AutoResetGymAgent, GymAgent

# Not present in the A2C version...
from salina.logger import TFLogger

def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v
  
class ProbAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)
        
class ActionAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)
        
class CriticAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic", t), critic)
        
def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
  
class EnvAgent(GymAgent):
  # Create the environment agent
  # This agent implements N gym environments with auto-reset
  def __init__(self, cfg):
    super().__init__(
      get_class(cfg.algorithm.env),
      get_arguments(cfg.algorithm.env),
      n_envs=cfg.algorithm.n_envs,
    )
    self.env = instantiate_class(cfg.algorithm.env)

  # TODO: replace the code below by a unique context-sensitive function 
  # that returns self.action_space.shape[0] or self.action_space.n
  # depending on whether the action space is a Box or a Discrete space


  # This is necessary to create the corresponding RL agent
  def get_obs_and_actions_sizes(self):
    if self.action_space.isinstance(gym.spaces.Box):
        # Return the size of the observation and action spaces of the environment
        # In the case of a continuous action environment
        return self.observation_space.shape[0], self.action_space.shape[0]
    elif self.action_space.isinstance(gym.spaces.Discrete):
        # Return the size of the observation and action spaces of the environment
      return self.observation_space.shape[0], self.action_space.n
    else:
      print ("unknown type of action space", self.action_space)
      return None
    
class A2CParamAgent(Agent):
  def __init__(self, cfg, env_agent):
    super().__init__()
    # TODO créer un dictionnaire d'hyper-params et leur tirer des valeurs initiales dans des intervalles
    self.hyperparam = { 
    "discount_factor" : random.uniform(0.0,1.0), #tire un float aléatoire entre 0 et 1
    "n_timesteps" : random.randint(1,20),  # tire un int random entre 1 et 20
    "entropy_coef" : random.uniform(0.0,1.0), # tire un random entre 0 et 1
    "critic_coef" : random.uniform(0.0,1.0), # tire un random entre 0 et 1
    "a2c_coef" : random.uniform(0.0,1.0) # tirer un random entre 0 et 1
    }

    observation_size,  n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env
    self.prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )
    action_agent = ActionAgent()
    self.critic_agent = CriticAgent(
      observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    # Combine env and policy agents
    agent = Agents(env_agent, self.prob_agent, action_agent)
    # Get an agent that is executed on a complete workspace
    self.agent = TemporalAgent(agent)
    self.agent.seed(cfg.algorithm.env_seed)

  def get_a2c_agent(self):
    return self.agent

  def get_a2c_agent_and_others(self):
    return self.agent, self.prob_agent, self.critic_agent

  def compute_critic_loss(self, reward, done, critic):
    # Compute temporal difference
    target = reward[1:] + self.hyperparam["discount_factor"] * critic[1:].detach() * (1 - done[1:].float())
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td ** 2
    critic_loss = td_error.mean()
    return critic_loss, td

  def compute_a2c_loss(action_probs, action, td):
    action_logp = _index(action_probs, action).log()
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()

  def copy(self, param_agent):
    # aussi copier les hyper-params
    self.agent = copy(param_agent.get_a2c_agent()) #écrire la fonction de copie
    self.agent.hyperparam = param_agent.hyperparam

  def mutate(self, cfg):
    # aussi muter les hyper-params
    # self.agent.mutate(cfg.mutation_rate) #écrire la fonction mutate
    for i in self.agent.hyperparam:
      self.agent.hyperparam[i] = random.uniform(self.agent.hyperparam[i], self.agent.hyperparam[i] * cfg.mutation_rate) #il n'y a pas de mutation_rate dans le cfg
    
    
class Logger():

  def __init__(self, cfg):
    self.logger = instantiate_class(cfg.logger)

  def add_log(self, log_string, loss, epoch):
    self.logger.add_scalar(log_string, loss.item(), epoch)

  # Log losses
def log_losses(self, cfg, epoch, critic_loss, entropy_loss, a2c_loss):
    self.add_log("critic_loss", critic_loss, epoch)
    self.add_log("entropy_loss", entropy_loss, epoch)
    self.add_log("a2c_loss", a2c_loss, epoch)

    
    # Configure the optimizer over the a2c agent
  def setup_optimizers(cfg, prob_agent, critic_agent):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer
  
def execute_agent(cfg, epoch, workspace, agent):
  if epoch > 0:
      workspace.zero_grad()
      workspace.copy_n_last_steps(1)
      agent(
        workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True)
  else:
    agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True)
    
class PBT():
  def sort_perf(liste):
    #trie les agents de la liste par ordre de performance décroissante
    pass

  def random_select(index, liste):
    # renvoie un élément de la liste entre 0 et index
    pass

# local training loop
# en faire une méthode de a2c_agent...

def train_a2c_agent(cfg, param_agent, logger):
  a2c_agent, prob_agent, critic_agent = param_agent.get_a2c_agent_and_others()

  tcritic_agent = TemporalAgent(critic_agent)

  workspace = salina.Workspace()

  # en faire une méthode?
  optimizer = setup_optimizers(cfg, prob_agent, critic_agent)
  epoch = 0
  for epoch in range(cfg.algorithm.max_epochs):
    # Execute the agent in the workspace
    execute_agent(cfg, epoch, workspace, a2c_agent)

    # Compute the critic value over the whole workspace
    tcritic_agent(workspace, n_steps=a2c_agent.hyperparam["n_timesteps"])

    # Get relevant tensors (size are timestep x n_envs x ....)
    critic, done, action_probs, reward, action = workspace[
        "critic", "env/done", "action_probs", "env/reward", "action"
      ]

    # Compute critic loss
    critic_loss, td = param_agent.compute_critic_loss(reward, done, critic)

    # Compute entropy loss
    entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

    # Compute A2C loss
    a2c_loss = param_agent.compute_a2c_loss(action_probs, action, td)

    # Store the losses for tensorboard display
    logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

    # Compute the total loss
    loss = (
      - param_agent.hyperparam["entropy_coef"] * entropy_loss
      + param_agent.hyperparam["critic_coef"] * critic_loss
      - param_agent.hyperparam["a2c_coef"] * a2c_loss
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute the cumulated reward on final_state
    creward = workspace["env/cumulated_reward"]
    creward = creward[done]
    if creward.size()[0] > 0:
      logger.add_log("reward", creward.mean(), epoch)
      
def run_pbt_a2c(cfg):
  # 1)  Build the  logger

  pbt_agent = PBT()
  logger = Logger(cfg)
  
  # 2) Create the environment agent
  env_agent = EnvAgent(cfg)

  budget = cfg.algorithm.budget
  pop_size = cfg.algorithm.pop_size

  pop = []

  for i in range(pop_size):
    a2c_param_agent = A2CParamAgent(cfg, env_agent) #faut-il un env_agent pour chaque agent a2c?
    pop.append(a2c_param_agent)

  while (budget > 0):

    for i in range(pop_size):
      train_a2c_agent(cfg, pop[i], logger)
  
    pbt_agent.sort_perf(pop) # tri par performance décroissante
    for j in range(pop_size*0.8, pop_size):
      source = pbt_agent.random_select(pop_size*0.2, pop)
      pop[j].copy(source)
      pop[j].mutate(cfg) #ajouter le mutation rate

    budget = budget - 1
