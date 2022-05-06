from ftplib import all_errors
import json
import sys
import torch
from matplotlib import pyplot as plt, ticker
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

class CustomLogger:
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
                "reward": float(crewards[a]),
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
    
    def get_all_hyperparam_values(self, hyperparam):
        all_values = torch.tensor([])
        for timestep in self.data.keys():
            timestep_values = torch.tensor([agent['hyperparameters'][hyperparam] for agent in self.data[timestep].values()])
            all_values=torch.cat((all_values, timestep_values.unsqueeze(0)))
        return all_values

    def start_plot(self):
        plt.close()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10,6)
        self.ax.get_xaxis().set_major_locator(ticker.AutoLocator())
        self.ax.grid()
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    def end_plot(self, file):
        plt.savefig(file)

    def plot_rewards_mean_and_individuals(self, line_color='blue'):
        timesteps = self.data.keys()
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([min(0, all_rewards.min().item()), max(0, all_rewards.max().item())])
        mean_rewards = all_rewards.mean(1)
        agents = range(all_rewards.size(1))
        timesteps = [int(t) / len(agents) for t in self.data.keys()]  # Adjust the timesteps, to show the number of timesteps that each agent has done
        for a in agents:
            plt.plot(timesteps, all_rewards.select(1, a), color='grey', alpha=0.3, linewidth=1)
        #plt.scatter(timesteps, mean_rewards, color=line_color)
        plt.plot(timesteps, mean_rewards, color=line_color, linewidth=2)
        self.ax.set(xlabel='timestep', ylabel='reward', title='Évolutions individuelles de la récompense')

    def plot_rewards_mean_and_std(self, line_color='blue'):
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([min(0, all_rewards.min().item()), max(0, all_rewards.max().item())])
        mean_rewards = all_rewards.mean(1)
        std_rewards = all_rewards.std(1)
        agents = range(all_rewards.size(1))
        timesteps = [int(t) / len(agents) for t in self.data.keys()]  # Adjust the timesteps, to show the number of timesteps that each agent has done
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color=line_color, alpha=0.5)
        #plt.scatter(timesteps, mean_rewards, color=line_color)
        plt.plot(timesteps, mean_rewards, color=line_color, linewidth=2)
        self.ax.set(xlabel='timestep', ylabel='reward', title='Moyenne et écart type des récompenses')
    
    def plot_hyperparam_individuals(self, hyperparam, line_color='blue'):
        all_values = self.get_all_hyperparam_values(hyperparam)
        self.ax.set_ylim([0, all_values.max().item()])
        agents = range(all_values.size(1))
        timesteps = [int(t) / len(agents) for t in self.data.keys()]  # Adjust the timesteps, to show the number of timesteps that each agent has done
        mean_values = all_values.mean(1)
        for a in agents:
            plt.plot(timesteps, all_values.select(1, a), color='grey', alpha=0.3, linewidth=1)
        #plt.scatter(timesteps, mean_values, color=line_color)
        plt.plot(timesteps, mean_values, color=line_color, linewidth=2)
        self.ax.set(xlabel='timestep', ylabel=hyperparam, title='Évolutions individuelles de ' + hyperparam)

    def plot_hyperparam_mean_and_std(self, hyperparam, line_color='blue'):
        all_values = self.get_all_hyperparam_values(hyperparam)
        self.ax.set_ylim([0, all_values.max().item()])
        mean_values = all_values.mean(1)
        std_values = all_values.std(1)
        agents = range(all_values.size(1))
        timesteps = [int(t) / len(agents) for t in self.data.keys()]  # Adjust the timesteps, to show the number of timesteps that each agent has done
        plt.fill_between(timesteps, mean_values - std_values, mean_values + std_values, color=line_color, alpha=0.5)
        #plt.scatter(timesteps, mean_values, color=line_color)
        plt.plot(timesteps, mean_values, color=line_color, linewidth=2)
        self.ax.set(xlabel='timestep', ylabel=hyperparam, title='Moyenne et écart type de ' + hyperparam)


if __name__ == "__main__":
    size = int(sys.argv[1])

    logger = CustomLogger()
    logger.open('./output_size_{}.json'.format(size))

    output_dir = './graphs/size_{}/'.format(size)

    # Graphs in PNG
    logger.start_plot()
    logger.plot_rewards_mean_and_individuals()
    logger.end_plot(output_dir + 'reward_individuals.png')
    logger.start_plot()
    logger.plot_rewards_mean_and_std()
    logger.end_plot(output_dir + 'reward_std.png')
    logger.start_plot()
    logger.plot_hyperparam_individuals('a2c_coef', line_color='green')
    logger.end_plot(output_dir + 'a2c_coef_inds.png')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('a2c_coef', line_color='green')
    logger.end_plot(output_dir + 'a2c_coef_mean_and_std.png')
    logger.start_plot()
    logger.plot_hyperparam_individuals('critic_coef', line_color='red')
    logger.end_plot(output_dir + 'critic_coef_inds.png')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('critic_coef', line_color='red')
    logger.end_plot(output_dir + 'critic_coef_mean_and_std.png')
    logger.start_plot()
    logger.plot_hyperparam_individuals('entropy_coef', line_color='orange')
    logger.end_plot(output_dir + 'entropy_coef_inds.png')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('entropy_coef', line_color='orange')
    logger.end_plot(output_dir + 'entropy_coef_mean_and_std.png')

    # Graphs in PDF, for LaTeX
    logger.start_plot()
    logger.plot_rewards_mean_and_individuals()
    logger.end_plot(output_dir + 'reward_individuals.pdf')
    logger.start_plot()
    logger.plot_rewards_mean_and_std()
    logger.end_plot(output_dir + 'reward_std.pdf')
    logger.start_plot()
    logger.plot_hyperparam_individuals('a2c_coef', line_color='green')
    logger.end_plot(output_dir + 'a2c_coef_inds.pdf')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('a2c_coef', line_color='green')
    logger.end_plot(output_dir + 'a2c_coef_mean_and_std.pdf')
    logger.start_plot()
    logger.plot_hyperparam_individuals('critic_coef', line_color='red')
    logger.end_plot(output_dir + 'critic_coef_inds.pdf')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('critic_coef', line_color='red')
    logger.end_plot(output_dir + 'critic_coef_mean_and_std.pdf')
    logger.start_plot()
    logger.plot_hyperparam_individuals('entropy_coef', line_color='orange')
    logger.end_plot(output_dir + 'entropy_coef_inds.pdf')
    logger.start_plot()
    logger.plot_hyperparam_mean_and_std('entropy_coef', line_color='orange')
    logger.end_plot(output_dir + 'entropy_coef_mean_and_std.pdf')