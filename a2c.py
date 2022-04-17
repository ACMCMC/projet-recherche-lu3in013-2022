import multiprocessing
from copy import deepcopy
from typing import Union

import hydra
import omegaconf
import salina
import torch
import torch.nn as nn
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, TemporalAgent
from salina.agents.asynchronous import AsynchronousAgent

from common import get_cumulated_reward
from env import AutoResetEnvAgent
from utils import build_nn
from plot import CrewardsLogger, plot_hyperparams, Logger


class CriticAgent(salina.TAgent):
    def __init__(self, observation_size, hidden_layer_sizes):
        super().__init__()
        # Create the critic neural network
        # We use a function that takes a list of layer sizes and returns the neural network
        # TODO: This is for continuous actions, but we should be able to use it for discrete actions as well
        self.critic_model = build_nn([observation_size] + list(hidden_layer_sizes) + [1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        critic = self.critic_model(input).squeeze(-1)
        self.set(("critic", t), critic)

class A2CAgent(salina.TAgent):
    '''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''

    def __init__(self, parameters, observation_size, hidden_layer_sizes, action_size, stochastic=True, std_param=None, discount_factor=0.95):
        super().__init__()
        # Create the action neural network
        # We use a function that takes a list of layer sizes and returns the neural network
        self.action_model = build_nn([observation_size] + list(hidden_layer_sizes) + [action_size], output_activation=nn.Tanh, activation=nn.ReLU)
        # Create the critic neural network

        self.observation_size = observation_size # The size of the observations
        self.hidden_layer_sizes = hidden_layer_sizes # The sizes of the hidden layers
        self.action_size = action_size # The size of the actions (output of the action model)
        self.stochastic = stochastic # Whether to use stochastic or deterministic actions
        self.discount_factor = discount_factor # The discount factor for the temporal difference
        self.params = omegaconf.DictConfig(content=parameters) # The hyperparameters of the agent
        if std_param is None:
            init_variance = torch.randn(action_size, 1)
            self.std_param = nn.parameter.Parameter(init_variance)
        else:
            self.std_param = std_param.clone()
        self.softplus = torch.nn.Softplus() # Like ReLU, but with a smooth gradient at zero

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        scores = self.action_model(input)
        dist = torch.distributions.Normal(scores, self.softplus(self.std_param))
        self.set(("entropy", t), dist.entropy())

        if self.stochastic:
            action = torch.tanh(dist.sample())
        else:
            action = torch.tanh(scores)

        logprobs = dist.log_prob(action).sum(axis=-1)

        self.set(("action", t), action)
        self.set(("action_logprobs", t), logprobs)

    @staticmethod
    def _index_3d_tensor_with_2d_tensor(tensor_3d: torch.Tensor, tensor_2d: torch.Tensor):
        # TODO: What is the purpose of this function?
        x, y, z = tensor_3d.size()
        t = tensor_3d.reshape(x*y, z)
        tt = tensor_2d.reshape(x*y)
        v = t[torch.arrange(x*y), tt]
        v = v.reshape(x, y)
        return v
    
    def compute_a2c_loss(self, action_logprobs: torch.Tensor, td: float) -> float:
        a2c_loss = action_logprobs[:-1] * td.detach() # TODO: Is it OK to calculate it like this?
        return a2c_loss.mean()

    def get_hyperparameter(self, param_name):
        return self.params[param_name]

    def set_hyperparameter(self, param_name, value):
        self.params[param_name] = value

    def clone(self):
        new = A2CAgent(self.params, self.observation_size, self.hidden_layer_sizes, self.action_size, self.stochastic, std_param=self.std_param)
        return new

    def compute_critic_loss(self, reward, done, critic) -> Union[float, float]:
        # Compute temporal difference
        target = reward[1:] + self.discount_factor * critic[1:].detach() * (1 - done[1:].float())
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()
        return critic_loss, td
    
    def compute_loss(self, workspace: Workspace, timestep, logger) -> float:
        critic, done, action_logprobs, reward, action, entropy = workspace[
            "critic", "env/done", "action_logprobs", "env/reward", "action", "entropy"
        ]

        critic_loss, td = self.compute_critic_loss(reward, done, critic)

        entropy_loss = entropy.mean()

        a2c_loss = self.compute_a2c_loss(action_logprobs, td)

        logger.log_losses(timestep, critic_loss, entropy_loss, a2c_loss)

        loss = (
            - self.params.entropy_coef * entropy_loss
            + self.params.critic_coef * critic_loss
            - self.params.a2c_coef * a2c_loss
        )

        return loss


class A2CParameterizedAgent(salina.TAgent):
    '''
    This agent is in charge of mutating the hyperparameters of an A2C agent.
    '''

    def __init__(self, parameters, observation_size, hidden_layer_sizes, action_size, mutation_rate, stochastic=True, discount_factor=0.95):
        super().__init__()

        self.mutation_rate = mutation_rate
        
        simplified_parameters = omegaconf.DictConfig(content={}) # The A2C Agent only sees a dictionary of (param_name, param_value) entries
        self.params_metadata = omegaconf.DictConfig(content={}) # This wrapper will store the metadata for the parameters of the A2C agent, so it knows how to change them when needed
        for param in parameters:
            generated_val = torch.distributions.Uniform(parameters[param].min, parameters[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            self.params_metadata[param] = {'min': parameters[param].min, 'max': parameters[param].max}
            simplified_parameters[param] = generated_val

        self.a2c_agent = A2CAgent(parameters=simplified_parameters, observation_size=observation_size, hidden_layer_sizes=hidden_layer_sizes, action_size=action_size, stochastic=stochastic, discount_factor=discount_factor)
        
    def mutate_hyperparameters(self):
        '''This function mutates, randomly, all the hyperparameters of this agent, according to the mutation rate'''
        for param in self.params_metadata:
            # We'll generate a completely random value, and mutate the original one according to the mutation rate.
            old_val = self.a2c_agent.get_hyperparameter(param)
            generated_val = torch.distributions.Uniform(self.params_metadata[param].min, self.params_metadata[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            discriminator = torch.distributions.Uniform(0, 1).sample().item()
            if discriminator > 0.5:
                mutated_val = 0.8 * old_val
            else:
                mutated_val = 1.2 * old_val
            self.a2c_agent.set_hyperparameter(param, mutated_val)
    
    def get_agent(self):
        return self.a2c_agent

    def compute_critic_loss(self, reward, done, critic):
        return self.a2c_agent.compute_critic_loss(reward, done, critic)
    
    def compute_a2c_loss(self, action_probs, action, td):
        return self.a2c_agent.compute_a2c_loss(action_probs, action)

    def copy(self, other):
        self.a2c_agent = deepcopy(other.get_agent())

    def get_cumulated_reward(self):
        return self.a2c_agent.get_cumulated_reward()

    def __call__(self, workspace, t, **kwargs):
        return self.a2c_agent(t=t, workspace=workspace, kwargs=kwargs)
    
    def compute_loss(self, **kwargs):
        return self.a2c_agent.compute_loss(**kwargs)

def create_agent(cfg):
    environment = AutoResetEnvAgent(cfg)
    # observation_size: the number of features of the observation (in Pendulum-v1, it is 3)
    observation_size = environment.get_observation_size()
    # hidden_layer_size: the number of neurons in the hidden layer
    hidden_layer_sizes = list(cfg.algorithm.neural_network.hidden_layer_sizes)
    # action_size: the number of parameters to output as actions (in Pendulum-v1, it is 1)
    action_size = environment.get_action_size()

    workspace = Workspace()

    # The agent that we'll train will use the A2C algorithm
    a2c_agent = A2CParameterizedAgent(cfg.algorithm.hyperparameters, observation_size, hidden_layer_sizes, action_size, cfg.algorithm.mutation_rate, discount_factor=cfg.algorithm.discount_factor)
    # To generate the observations, we need a gym agent, which will provide the values in 'env/env_obs'
    environment_agent = AutoResetEnvAgent(cfg)
    temporal_agent = TemporalAgent(Agents(environment_agent, a2c_agent))
    temporal_agent.seed(cfg.algorithm.stochasticity_seed)

    return temporal_agent, workspace

def a2c_train(cfg, agent, workspace, logger):
    epoch_logger = CrewardsLogger()

    total_timesteps = 0

    five_last_rewards = torch.tensor([])

    # Configure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.optimizer)
    optimizer = get_class(cfg.optimizer)(
        agent.parameters(), **optimizer_args
    )

    for epoch in range(cfg.algorithm.max_epochs):
        print("Epoch: {}".format(epoch))
        consumed_budget = 0
        while consumed_budget < cfg.algorithm.train_budget:
            if consumed_budget > 0:
                workspace.zero_grad()
                workspace.copy_n_last_steps(1)
                agent(t=1, workspace=workspace, n_steps=cfg.algorithm.num_timesteps - 1)
            else:
                agent(t=0, workspace=workspace, n_steps=cfg.algorithm.num_timesteps)


            steps = (workspace.time_size() - 1) * workspace.batch_size()
            consumed_budget += steps
            
            done = workspace["env/done"]

            loss = agent.agent[1].compute_loss(workspace=workspace, timestep=total_timesteps + consumed_budget, logger=logger)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            creward = workspace["env/cumulated_reward"]
            creward = creward[done]

            five_last_rewards = torch.cat((five_last_rewards, creward))[-5:]

            logger.add_log("reward", five_last_rewards.mean(), total_timesteps + consumed_budget)
        
        total_timesteps += consumed_budget

        # They have all finished executing
        print('Finished epoch {}'.format(epoch))
        
        #cumulated_reward = five_last_rewards.mean().item()
        
        #epoch_logger.log_epoch(total_timesteps, torch.tensor([cumulated_reward]))
        #plot_hyperparams([agent.agent[1].get_agent()])


@hydra.main(config_path=".", config_name="pbt.yaml")
def run_a2c(cfg):
    # First, build the  logger
    logger = Logger(cfg)
    agent, workspace = create_agent(cfg)
    a2c_train(cfg, agent, workspace, logger=logger)

if __name__ == '__main__':
    run_a2c()
