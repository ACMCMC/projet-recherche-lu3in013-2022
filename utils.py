from torch import nn
import torch

def build_nn(layers_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """
    This function builds a neural network from a list of layer sizes.
    Parameters:
        layers_sizes: A list of integers. Each integer represents the size of a layer.
        activation: The activation function to use in each layer.
        output_activation: The activation function to use in the output layer.
    """
    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        if i < len(layers_sizes) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)

def save_model(self, filename) -> None:
        """
        Save a neural network model into a file
        :param filename: the filename, including the path
        :return: nothing
        """
        torch.save(self, filename)

def load_model(filename):
    """
    Load a neural network model from a file
    :param filename: the filename, including the path
    :return: the resulting pytorch network
    """
    net = torch.load(filename)
    return net

def gae(critic, reward, must_bootstrap, discount_factor, gae_coef):
    mb = must_bootstrap.float()
    td = reward[:-1] + discount_factor * critic[1:].detach() * mb - critic[:-1]
    # handling td0 case
    if gae_coef == 0.0:
        return td

    td_shape = td.shape[0]
    gae_val = td[-1]
    gae_vals = [gae_val]
    for t in range(td_shape - 2, -1, -1):
        gae_val = td[t] + discount_factor * gae_coef * mb[:-1][t] * gae_val
        gae_vals.append(gae_val)
    gae_vals = list([g.unsqueeze(0) for g in reversed(gae_vals)])
    gae_vals = torch.cat(gae_vals, dim=0)
    return gae_vals