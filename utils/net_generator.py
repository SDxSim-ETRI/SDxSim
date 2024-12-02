import torch.nn as nn

def create_network(input_dim, output_dim, hidden_layers, activation_fn):
    layers = []
    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    for i in range(len(layer_sizes) - 2):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation_fn())
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))  # 마지막 출력 레이어

    print(nn.Sequential(*layers))    
    return nn.Sequential(*layers)

def get_activation_fn(name):
    activation_map = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid
    }
    return activation_map[name]