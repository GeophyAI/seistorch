import torch
from torch.nn.functional import conv2d

def _laplacian(y, h):
    """Laplacian operator"""
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(y.device)
    operator = h ** (-2) * kernel
    y = y.unsqueeze(1)
    # y = pad(y,pad=(0,0,1,1), mode='circular')
    # y = pad(y,pad=(1,1,0,0),mode='circular')
    return conv2d(y, operator, padding=1).squeeze(1)


def _time_step(b, c, y1, y2, dt, h):
    # Equation S8(S9)
    # When b=0, without boundary conditon.
    y = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                (2 / dt**2 * y1 - torch.mul((dt**-2 - b * dt**-1), y2)
                + torch.mul(c.pow(2), _laplacian(y1, h)))
                )
    return y

# Define input tensors

b = torch.rand((10,10), requires_grad=True)
c = torch.rand((10,10), requires_grad=True)
y1 = torch.rand((1,10,10), requires_grad=True)
y2 = torch.rand((1,10,10), requires_grad=True)
dt = torch.tensor(0.1)
h = torch.tensor(0.01)

def compute_segment_loss(y1, y2, b, c, dt, h, start_t, end_t, observed_data):
    loss = 0
    for t in range(start_t, end_t):
        y = _time_step(b, c, y1, y2, dt, h)
        mse_loss = torch.mean((y - observed_data[t]) ** 2)
        loss += mse_loss
        y2 = y1
        y1 = y
    return loss

def gradient_checkpointing(b, c, y1, y2, dt, h, num_timesteps, observed_data, num_segments):
    segment_size = num_timesteps // num_segments
    total_loss = 0
    for i in range(num_segments):
        start_t = i * segment_size
        end_t = (i + 1) * segment_size if i != num_segments - 1 else num_timesteps

        # Calculate the loss for the segment
        segment_loss = compute_segment_loss(y1, y2, b, c, dt, h, start_t, end_t, observed_data)

        # Accumulate the total loss
        total_loss += segment_loss

        # Calculate gradients for the segment
        segment_grads = torch.autograd.grad(segment_loss, (b, c, y1, y2, dt, h), retain_graph=True)

        # Accumulate the gradients
        if i == 0:
            grads = [g.clone() for g in segment_grads]
        else:
            grads = [grads[i] + g.clone() for i, g in enumerate(segment_grads)]

        # Update y1 and y2 for the next segment
        y2 = y1
        y1 = segment_grads[2]

    return total_loss, grads

# Set the number of timesteps and segments
num_timesteps = 10
num_segments = 2

# Create a dummy observed_data tensor
observed_data = [torch.randn(10,10) for _ in range(num_timesteps)]

# Compute total loss and gradients using gradient checkpointing
total_loss, gradients = gradient_checkpointing(b, c, y1, y2, dt, h, num_timesteps, observed_data, num_segments)

# Check total loss and gradients
print("Total Loss:", total_loss.item())
print("Gradients:")
print("b.grad: ", gradients[0])
print("c.grad: ", gradients[1])
print("y1.grad: ", gradients[2])
print("y2.grad: ", gradients[3])
print("dt.grad: ", gradients[4])