import torch

# TODO:
# 1. 10.1190/GEO2014-0542.1 random boundary condition with random shape

def random_fill(data, num_layers, mind, maxd, multiple=False):
    # Get the shape of the data
    shape = tuple(data.shape)
    mind = int(mind)
    maxd = int(maxd)
    # Generate random values
    random_values = torch.randint(mind, maxd + 1, size=shape).to(data)

    # Create a new tensor to hold the padded data
    padded_data = random_values.clone()
    # Copy the data into the padded tensor
    top = 0 if multiple else num_layers
    if len(shape) == 2:  # for 2D data
        data = data[top:-num_layers, num_layers:-num_layers]
        padded_data[top:-num_layers, num_layers:-num_layers] = data
    elif len(shape) == 3:  # for 3D data
        padded_data[num_layers:-num_layers, num_layers:-num_layers, num_layers:-num_layers] = data
    else:
        raise ValueError("Unsupported data dimension. Only 2D and 3D data are supported.")

    return padded_data