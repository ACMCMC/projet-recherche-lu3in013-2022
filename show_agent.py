import sys
import hydra
from pbt import PBTAgent
from utils import load_model

AGENT_PATH = '/home/acmc/repos/projet-recherche-lu3in013-2022/saved_agents/agent_71.pickle'

@hydra.main(config_path=".", config_name="pbt.yaml")
def show_agent(cfg):
    agent = PBTAgent(cfg)
    agent.load(AGENT_PATH, cfg)
    simulated_workspace = agent.run_simulation()
    renders = simulated_workspace['env/rendering']
    import matplotlib.pyplot as plt
    for (i, render) in enumerate(renders):
        nparray = render.numpy()
        plt.imsave('/home/acmc/repos/projet-recherche-lu3in013-2022/renders/{}.png'.format(i), nparray)

if __name__ == "__main__":
    show_agent()