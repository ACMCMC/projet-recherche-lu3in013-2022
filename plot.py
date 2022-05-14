from datetime import datetime
from ftplib import all_errors
import json
import os
import re
import sys
import matplotlib
import torch
from matplotlib import pyplot as plt, ticker
from salina import instantiate_class
import numpy

ALPHA_STD = 0.3
ALPHA_INDIVIDUALS = 0.3
LINEWIDTH_INDIVIDUALS = 0.75
LINEWIDTH_MEAN = 2.0


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
    # Will contain the hyperparameters of each agent, in the form {'a2c_coef': [0.8, 0.7, ...], 'gamma': [0.2, 0.1, ...], ...}
    hyperparams = {}
    for hyperparam in agents_list[0].params.keys():
        # Put an empty tensor in the dictionary for each hyperparam, to be filled later
        hyperparams[hyperparam] = torch.tensor([])
    for hyperparam in hyperparams.keys():
        for agent in agents_list:
            hyperparams[hyperparam] = torch.cat(
                (hyperparams[hyperparam], torch.tensor([agent.params[hyperparam]])))
    plt.boxplot(hyperparams.values())
    plt.xticks(range(1, len(hyperparams) + 1), hyperparams.keys())
    plt.savefig('/home/diego/projet-recherche-lu3in013-2022/hyperparams.png')


class CustomLogger:
    def __init__(self, config=None) -> None:
        self.data = {}
        self.dir_location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        if config:
            self.environment_name = config.env.env_name
            self.population_size = config.algorithm.population_size
            self.output_location = os.path.join(self.dir_location, 'raw_data')
            self.output_filename = os.path.join(self.output_location, 'output_{env}_{size}_{time}.json'.format(
                env=self.environment_name, size=self.population_size, time=datetime.now().strftime("%d-%m_%H:%M:%S")))

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
        with open(self.output_filename, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def open(self, file):
        with open(os.path.join(self.dir_location, file), 'r') as handle:
            self.data = json.load(handle)

    def get_all_rewards(self):
        all_rewards = torch.tensor([])
        for timestep in self.data.keys():
            timestep_rewards = torch.tensor(
                [agent['reward'] for agent in self.data[timestep].values()])
            all_rewards = torch.cat(
                (all_rewards, timestep_rewards.unsqueeze(0)))
        return all_rewards

    def get_all_hyperparam_values(self, hyperparam):
        all_values = torch.tensor([])
        for timestep in self.data.keys():
            timestep_values = torch.tensor(
                [agent['hyperparameters'][hyperparam] for agent in self.data[timestep].values()])
            all_values = torch.cat((all_values, timestep_values.unsqueeze(0)))
        return all_values

    def start_plot(self):
        plt.close()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 6)
        self.ax.get_xaxis().set_major_locator(ticker.AutoLocator())
        self.ax.grid()
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        matplotlib.rc('font',
        #    family='DejaVu Sans',
        #    weight='bold',
            size=16)
        #plt.xlabel('xlabel', fontsize=14)
        #plt.ylabel('ylabel', fontsize=14)

    def end_plot(self, file):
        plt.savefig(file)

    def plot_rewards_mean_and_individuals(self, line_color='blue'):
        timesteps = self.data.keys()
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([min(0, all_rewards.min().item() * 1.1),
                         max(0, all_rewards.max().item() * 1.1)])
        mean_rewards = all_rewards.mean(1)
        agents = range(all_rewards.size(1))
        # Adjust the timesteps, to show the number of timesteps that each agent has done
        timesteps = [int(t) / len(agents) for t in self.data.keys()]
        for a in agents:
            plt.plot(timesteps, all_rewards.select(1, a), color='grey',
                     alpha=ALPHA_INDIVIDUALS, linewidth=LINEWIDTH_INDIVIDUALS)
        #plt.scatter(timesteps, mean_rewards, color=line_color)
        plt.plot(timesteps, mean_rewards, color=line_color,
                 linewidth=LINEWIDTH_MEAN)
        self.ax.set(xlabel='timestep', ylabel='reward',
                    title='Évolutions individuelles de la récompense')

    def plot_rewards_mean_and_std(self, line_color='blue'):
        all_rewards = self.get_all_rewards()
        self.ax.set_ylim([min(0, all_rewards.min().item() * 1.1),
                         max(0, all_rewards.max().item() * 1.1)])
        mean_rewards = all_rewards.mean(1)
        std_rewards = all_rewards.std(1)
        agents = range(all_rewards.size(1))
        # Adjust the timesteps, to show the number of timesteps that each agent has done
        timesteps = [int(t) / len(agents) for t in self.data.keys()]
        plt.fill_between(timesteps, mean_rewards - std_rewards,
                         mean_rewards + std_rewards, color=line_color, alpha=ALPHA_STD)
        #plt.scatter(timesteps, mean_rewards, color=line_color)
        plt.plot(timesteps, mean_rewards, color=line_color,
                 linewidth=LINEWIDTH_MEAN)
        self.ax.set(xlabel='timestep', ylabel='reward',
                    title='Moyenne et écart type des récompenses')

    def plot_hyperparam_individuals(self, hyperparam, line_color='blue'):
        all_values = self.get_all_hyperparam_values(hyperparam)
        self.ax.set_ylim([0, all_values.max().item() * 1.1])
        agents = range(all_values.size(1))
        # Adjust the timesteps, to show the number of timesteps that each agent has done
        timesteps = [int(t) / len(agents) for t in self.data.keys()]
        mean_values = all_values.mean(1)
        for a in agents:
            plt.plot(timesteps, all_values.select(1, a), color='grey',
                     alpha=ALPHA_INDIVIDUALS, linewidth=LINEWIDTH_INDIVIDUALS)
        #plt.scatter(timesteps, mean_values, color=line_color)
        plt.plot(timesteps, mean_values, color=line_color,
                 linewidth=LINEWIDTH_MEAN)
        self.ax.set(xlabel='timestep', ylabel=hyperparam,
                    title='Évolutions individuelles de ' + hyperparam)

    def plot_hyperparam_mean_and_std(self, hyperparam, line_color='blue'):
        all_values = self.get_all_hyperparam_values(hyperparam)
        self.ax.set_ylim([0, all_values.max().item() * 1.1])
        mean_values = all_values.mean(1)
        std_values = all_values.std(1)
        agents = range(all_values.size(1))
        # Adjust the timesteps, to show the number of timesteps that each agent has done
        timesteps = [int(t) / len(agents) for t in self.data.keys()]
        plt.fill_between(timesteps, mean_values - std_values,
                         mean_values + std_values, color=line_color, alpha=ALPHA_STD)
        #plt.scatter(timesteps, mean_values, color=line_color)
        plt.plot(timesteps, mean_values, color=line_color,
                 linewidth=LINEWIDTH_MEAN)
        self.ax.set(xlabel='timestep', ylabel=hyperparam,
                    title='Moyenne et écart type de ' + hyperparam)


class CombinedGraphMaker:
    def __init__(self):
        self.data = {}  # This is a dict of data from different graphs. It's formatted as follows {size : data}
        self.colors = {}
        self._color_generator = plt.cm.get_cmap('hsv', 6)
        self._color_generator_counter = 0

    def _get_next_color(self):
        self._color_generator_counter = (self._color_generator_counter + 1) % 6
        return self._color_generator(self._color_generator_counter)

    def load(self, file):
        with open(file, 'r') as myfile:
            json_data = json.load(myfile)
            # You get the size of the population from the first item of the json file
            size = len(list(json_data.items())[0][1])
            self.data[size] = json_data
            self.colors[size] = self._get_next_color()
        myfile.close()

    def start_plot(self):
        plt.close()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 6)
        self.ax.get_xaxis().set_major_locator(ticker.AutoLocator())
        self.ax.grid()
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        matplotlib.rc('font',
        #    family='DejaVu Sans',
        #    weight='bold',
            size= 16)
        #plt.xlabel('xlabel', fontsize=14)
        #plt.ylabel('ylabel', fontsize=14)

    def end_plot(self, file):
        plt.savefig(file)

    def get_all_rewards(self, data):
        all_rewards = torch.tensor([])
        for timestep in data.keys():
            timestep_rewards = torch.tensor(
                [agent['reward'] for agent in data[timestep].values()])
            all_rewards = torch.cat(
                (all_rewards, timestep_rewards.unsqueeze(0)))
        return all_rewards

    def get_all_hyperparam_values(self, data, hyperparam):
        all_values = torch.tensor([])
        for timestep in data.keys():
            timestep_values = torch.tensor(
                [agent['hyperparameters'][hyperparam] for agent in data[timestep].values()])
            all_values = torch.cat((all_values, timestep_values.unsqueeze(0)))
        return all_values

    def plot_rewards_mean_and_individuals(self):
        all_rewards_all_sizes = torch.tensor([]) # This is a tensor of all rewards for all sizes, we will use this to know the max and min of the y axis
        for i in self.data.keys():
            data_i = self.data[i]
            timesteps = data_i.keys()
            all_rewards = self.get_all_rewards(data_i)
            mean_rewards = all_rewards.mean(1)
            agents = range(all_rewards.size(1))
            # Adjust the timesteps, to show the number of timesteps that each agent has done
            timesteps = [int(t) / len(agents) for t in data_i.keys()]
            for a in agents:
                plt.plot(timesteps, all_rewards.select(1, a), color='grey',
                         alpha=ALPHA_INDIVIDUALS, linewidth=LINEWIDTH_INDIVIDUALS)
                #plt.scatter(timesteps, mean_rewards, color=line_color)
            plt.plot(timesteps, mean_rewards,
                     color=self.colors[i], linewidth=LINEWIDTH_MEAN, label=i)
            all_rewards_all_sizes = torch.cat([all_rewards_all_sizes, all_rewards], dim=1)
        self.ax.set_ylim([min(0, all_rewards_all_sizes.min().item() * 1.1),
                         max(0, all_rewards_all_sizes.max().item() * 1.1)])
        plt.legend()
        self.ax.set(xlabel='timestep', ylabel='reward',
                    title='Combinaison des évolutions individuelles de la récompense')

    def plot_rewards_mean_and_std(self):
        all_rewards_all_sizes = torch.tensor([]) # This is a tensor of all rewards for all sizes, we will use this to know the max and min of the y axis
        for i in self.data.keys():
            data_i = self.data[i]
            all_rewards = self.get_all_rewards(data_i)
            mean_rewards = all_rewards.mean(1)
            std_rewards = all_rewards.std(1)
            agents = range(all_rewards.size(1))
            # Adjust the timesteps, to show the number of timesteps that each agent has done
            timesteps = [int(t) / len(agents) for t in data_i.keys()]
            plt.fill_between(timesteps, mean_rewards - std_rewards,
                             mean_rewards + std_rewards, color=self.colors[i], alpha=ALPHA_STD)
            #plt.scatter(timesteps, mean_rewards, color=line_color)
            plt.plot(timesteps, mean_rewards,
                     color=self.colors[i], linewidth=LINEWIDTH_MEAN, label=i)
            all_rewards_all_sizes = torch.cat([all_rewards_all_sizes, all_rewards], dim=1)
        self.ax.set_ylim([min(0, all_rewards_all_sizes.min().item() * 1.1),
                         max(0, all_rewards_all_sizes.max().item() * 1.1)])
        plt.legend()
        self.ax.set(xlabel='timestep', ylabel='reward',
                    title='Moyenne et écart type combinés des récompenses')

    def plot_hyperparam_individuals(self, hyperparam):
        for i in self.data.keys():
            data_i = self.data[i]
            all_values = self.get_all_hyperparam_values(data_i, hyperparam)
            #self.ax.set_ylim([0, all_values.max().item()])
            agents = range(all_values.size(1))
            # Adjust the timesteps, to show the number of timesteps that each agent has done
            timesteps = [int(t) / len(agents) for t in data_i.keys()]
            mean_values = all_values.mean(1)
            for a in agents:
                plt.plot(timesteps, all_values.select(1, a), color='grey',
                         alpha=ALPHA_INDIVIDUALS, linewidth=LINEWIDTH_INDIVIDUALS)
                #plt.scatter(timesteps, mean_values, color=line_color)
            plt.plot(timesteps, mean_values,
                     color=self.colors[i], linewidth=LINEWIDTH_MEAN, label=i)
        plt.legend()
        self.ax.set(xlabel='timestep', ylabel=hyperparam,
                    title='Combinaison des évolutions individuelles de ' + hyperparam)

    def plot_hyperparam_mean_and_std(self, hyperparam):
        for i in self.data.keys():
            data_i = self.data[i]
            all_values = self.get_all_hyperparam_values(data_i, hyperparam)
            #self.ax.set_ylim([0, all_values.max().item()])
            mean_values = all_values.mean(1)
            std_values = all_values.std(1)
            agents = range(all_values.size(1))
            # Adjust the timesteps, to show the number of timesteps that each agent has done
            timesteps = [int(t) / len(agents) for t in data_i.keys()]
            plt.fill_between(timesteps, mean_values - std_values, mean_values +
                             std_values, color=self.colors[i], alpha=ALPHA_STD)
            #plt.scatter(timesteps, mean_values, color=line_color)
            plt.plot(timesteps, mean_values,
                     color=self.colors[i], linewidth=LINEWIDTH_MEAN, label=i)
        plt.legend()
        self.ax.set(xlabel='timestep', ylabel=hyperparam,
                    title='Moyenne et écart type combinés de ' + hyperparam)


if __name__ == "__main__":
    # run:
    # python plot.py raw_data/output_CartPoleContinuous-v1_5_06-05_21:16:56.json raw_data/output_CartPoleContinuous-v1_10_06-05_22:24:13.json raw_data/output_CartPoleContinuous-v1_20_06-05_22:28:35.json raw_data/output_CartPoleContinuous-v1_40_06-05_22:29:37.json raw_data/output_CartPoleContinuous-v1_100_06-05_22:30:28.json
    # python plot.py raw_data/output_Pendulum-v1_5_06-05_21:24:38.json raw_data/output_Pendulum-v1_10_06-05_22:32:19.json raw_data/output_Pendulum-v1_20_06-05_22:33:41.json raw_data/output_Pendulum-v1_40_06-05_22:34:33.json raw_data/output_Pendulum-v1_100_06-05_22:35:12.json

    filelist = []
    for i in sys.argv[1:]:
        filelist.append(i)

    if len(sys.argv) == 6:
        combined_dirname = re.sub(
            r'(output_[^_]+_)\d+(.+)', r'\1all\2', filelist[0])

        output_dir = './graphs/{}/'.format(os.path.basename(combined_dirname))
        try:
            os.mkdir(output_dir)  # Create the directory if it does not exist
        except FileExistsError:
            pass

        graphmaker = CombinedGraphMaker()
        for i in filelist:
            graphmaker.load(i)

        # Graphs in PNG
        graphmaker.start_plot()
        graphmaker.plot_rewards_mean_and_individuals()
        graphmaker.end_plot(output_dir + 'reward_individuals.png')
        graphmaker.start_plot()
        graphmaker.plot_rewards_mean_and_std()
        graphmaker.end_plot(output_dir + 'reward_std.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('a2c_coef')
        graphmaker.end_plot(output_dir + 'a2c_coef_inds.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('a2c_coef')
        graphmaker.end_plot(output_dir + 'a2c_coef_mean_and_std.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('critic_coef')
        graphmaker.end_plot(output_dir + 'critic_coef_inds.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('critic_coef')
        graphmaker.end_plot(output_dir + 'critic_coef_mean_and_std.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('entropy_coef')
        graphmaker.end_plot(output_dir + 'entropy_coef_inds.png')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('entropy_coef')
        graphmaker.end_plot(output_dir + 'entropy_coef_mean_and_std.png')

        # Graphs in PDF, for LaTeX
        graphmaker.start_plot()
        graphmaker.plot_rewards_mean_and_individuals()
        graphmaker.end_plot(output_dir + 'reward_individuals.pdf')
        graphmaker.start_plot()
        graphmaker.plot_rewards_mean_and_std()
        graphmaker.end_plot(output_dir + 'reward_std.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('a2c_coef')
        graphmaker.end_plot(output_dir + 'a2c_coef_inds.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('a2c_coef')
        graphmaker.end_plot(output_dir + 'a2c_coef_mean_and_std.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('critic_coef')
        graphmaker.end_plot(output_dir + 'critic_coef_inds.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('critic_coef')
        graphmaker.end_plot(output_dir + 'critic_coef_mean_and_std.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_individuals('entropy_coef')
        graphmaker.end_plot(output_dir + 'entropy_coef_inds.pdf')
        graphmaker.start_plot()
        graphmaker.plot_hyperparam_mean_and_std('entropy_coef')
        graphmaker.end_plot(output_dir + 'entropy_coef_mean_and_std.pdf')

    for file in filelist:
        output_dir = './graphs/{}/'.format(os.path.basename(file))
        try:
            os.mkdir(output_dir)  # Create the directory if it does not exist
        except FileExistsError:
            pass

        logger = CustomLogger()
        logger.open(file)

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
        logger.plot_hyperparam_mean_and_std(
            'entropy_coef', line_color='orange')
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
        logger.plot_hyperparam_mean_and_std(
            'entropy_coef', line_color='orange')
        logger.end_plot(output_dir + 'entropy_coef_mean_and_std.pdf')
