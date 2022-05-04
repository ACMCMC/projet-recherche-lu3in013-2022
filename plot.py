from ftplib import all_errors
import json
import torch
from matplotlib import pyplot as plt
from salina import instantiate_class
import numpy

class Logger():
  def __init__(self, cfg):
    self.logger = instantiate_class(cfg.logger)

  def add_log(self, log_string, loss, epoch):
    self.logger.add_scalar(log_string, loss.item(), epoch)

  # Log losses
  def log_losses(self, epoch, critic_loss, entropy_loss, a2c_loss):
    self.add_log("critic_loss", critic_loss, epoch)
    self.add_log("entropy_loss", entropy_loss, epoch)
    self.add_log("a2c_loss", a2c_loss, epoch)

def plot_hyperparams(agents_list):
    plt.close()
    hyperparams = {} # Will contain the hyperparameters of each agent, in the form {'a2c_coef': [0.8, 0.7, ...], 'gamma': [0.2, 0.1, ...], ...}
    for hyperparam in agents_list[0].params.keys():
        hyperparams[hyperparam] = torch.tensor([]) # Put an empty tensor in the dictionary for each hyperparam, to be filled later
    for hyperparam in hyperparams.keys():
        for agent in agents_list:
            hyperparams[hyperparam] = torch.cat((hyperparams[hyperparam], torch.tensor([agent.params[hyperparam]])))
    plt.boxplot(hyperparams.values())
    plt.xticks(range(1, len(hyperparams) + 1), hyperparams.keys())
    plt.savefig('/home/acmc/repos/projet-recherche-lu3in013-2022/hyperparams.png')

class CrewardsLogger:
    def __init__(self) -> None:
        self.data = {}

    def log_epoch(self, timestep, crewards, agents):
        self.data[timestep] = {}
        for (i, a) in enumerate(agents):
            # Prepare the hyperparameters
            hyperparams = {}
            for hyperparam in a.action_agent.a2c_agent.params.keys():
                hyperparams[hyperparam] = float(
                    a.action_agent.a2c_agent.params[hyperparam])

            # Write the entry for the agent
            self.data[timestep][i] = {
                "reward": crewards[i].item(),
                "hyperparameters": hyperparams
            }
    
    def save(self):
        with open('/home/acmc/repos/projet-recherche-lu3in013-2022/output.json', 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def open(self, file):
        with open(file, 'r') as handle:
            self.data = json.load(handle)
    
    def get_all_rewards(self):
        all_rewards = torch.tensor([])
        for timestep in self.data.keys():
            timestep_rewards = torch.tensor([agent['reward'] for agent in self.data[timestep].values()])
            all_rewards=torch.cat((all_rewards, timestep_rewards.unsqueeze(0)))
        return all_rewards

    def start_plot(self):
        plt.close()
        self.fig, self.ax = plt.subplots()
    
    def end_plot(self, file):
        plt.savefig(file)

    def plot_rewards_mean_and_individuals(self):
        timesteps = self.data.keys()
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([0, all_rewards.max().item()])
        agents = range(all_rewards.size(1))
        mean_rewards = all_rewards.mean(1)
        plt.scatter(timesteps, mean_rewards, color='blue')
        plt.plot(timesteps, mean_rewards, color='blue')
        for a in agents:
            plt.plot(timesteps, all_rewards.select(1, a), alpha=0.3, color='black')
        self.ax.set(xlabel='timestep', ylabel='reward', title='Evolution of rewards')
        self.ax.grid()

    def plot_rewards_mean_and_std(self):
        timesteps = self.data.keys()
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([0, all_rewards.max().item()])
        mean_rewards = all_rewards.mean(1)
        std_rewards = all_rewards.std(1)
        plt.scatter(timesteps, mean_rewards, color='blue')
        plt.plot(timesteps, mean_rewards, color='blue')
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.5, color='blue')
        self.ax.set(xlabel='timestep', ylabel='reward', title='Evolution of rewards')
        self.ax.grid()

if __name__ == "__main__":
    logger = CrewardsLogger()
    logger.open('./output.json')
    logger.start_plot()
    logger.plot_rewards_mean_and_individuals()
    logger.end_plot('individuals.png')
    logger.start_plot()
    logger.plot_rewards_mean_and_std()
    logger.end_plot('rew_std.png')