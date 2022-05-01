import torch
from matplotlib import pyplot as plt
from salina import instantiate_class

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
        self.crewards: torch.Tensor = torch.tensor([]) # An empty tensor of the form ([ [],[],[] ], [ [],[],[] ], ...)

    def log_epoch(self, timestep, crewards):
        plt.close() # Clear the last figure
        mean_of_crewards = crewards.mean()
        std = crewards.std()


        tensor_to_cat = torch.tensor([timestep, mean_of_crewards, std]).unsqueeze(-1) # Gives us a tensor like [[timestep], [mean_of_crewards]] 
        self.crewards = torch.cat((self.crewards, tensor_to_cat), dim=1)
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim([self.crewards[1].min(0)[0].item(), 0])
        plt.scatter(self.crewards[0], self.crewards[1])
        plt.plot(self.crewards[0], self.crewards[1])
        plt.fill_between(self.crewards[0], self.crewards[1] - self.crewards[2], self.crewards[1] + self.crewards[2], alpha=0.5)
        self.ax.set(xlabel='timestep', ylabel='creward', title='Evolution of crewards')
        self.ax.grid()
        plt.savefig('/home/acmc/repos/projet-recherche-lu3in013-2022/fig.png')
    
    def show(self):
        plt.show()