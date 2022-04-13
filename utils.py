from torch import nn

def build_nn(layers_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        if i < len(layers_sizes) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)