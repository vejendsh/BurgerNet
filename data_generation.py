import torch
import torch.fft as fft

num_grid_points = 101
num_time_steps = 5
step_size = 0.0005
grid_cell_size = 1 / (num_grid_points - 1)
grid_points = torch.linspace(0, 1, num_grid_points)
nu = 0.1  # Kinematic Viscosity


# Function to generate frequency solutions
def gen_freq_sols(num_freq_sol=50000, lower_bound=-100, upper_bound=100, num_active_frequencies=1,
                  total_num_frequencies=51):
    # Generate random positions for the non-zero elements
    col_indices = torch.randint(0, total_num_frequencies, (num_freq_sol, num_active_frequencies))

    # Generate random complex numbers directly
    real_parts = torch.empty(num_freq_sol, num_active_frequencies).uniform_(lower_bound, upper_bound)
    imaginary_parts = torch.empty(num_freq_sol, num_active_frequencies).uniform_(lower_bound, upper_bound)
    complex_numbers = torch.complex(real_parts, imaginary_parts)

    # Initialize an empty tensor with the given shape
    freq_sol = torch.zeros((num_freq_sol, total_num_frequencies), dtype=torch.complex64)

    # Use advanced indexing to set the non-zero complex numbers in the tensor
    freq_sol[torch.arange(num_freq_sol).unsqueeze(1).repeat(1, num_active_frequencies), col_indices] = complex_numbers

    # freq_sol = fft.rfft(torch.abs(fft.irfft(freq_sol, num_grid_points)))

    return freq_sol


# Function to simulate time solution

def simulate(current_sol):
    old_sol = current_sol
    updated_sol = torch.empty_like(old_sol)
    con_1 = (step_size / grid_cell_size)
    con_2 = ((nu * step_size) / (grid_cell_size ** 2))

    # Iterating through the difference equation
    for _ in range(num_time_steps):
        updated_sol[:, 0] = (old_sol[:, 0] - old_sol[:, 0] * con_1 * (old_sol[:, 0] - old_sol[:, -2])
                             + con_2 * (old_sol[:, 1] + old_sol[:, -2] - (2 * old_sol[:, 0])))

        updated_sol[:, 1:-1] = (
                old_sol[:, 1:-1] - old_sol[:, 1:-1] * con_1 * (old_sol[:, 1:-1] - old_sol[:, :-2])
                + con_2 * (old_sol[:, 2:] + old_sol[:, :-2] - (2 * old_sol[:, 1:-1])))

        updated_sol[:, -1] = updated_sol[:, 0]  # Periodic Boundary Condition

        old_sol = updated_sol

    return updated_sol


# Function to Generate features from frequency solutions.

def gen_features(freq_sols):
    features = torch.stack((freq_sols.real, freq_sols.imag), dim=-1).view(freq_sols.shape[0], -1)

    return features


# Function to generate labels from features

def gen_labels(freq_sols):
    labels = fft.rfft(simulate(fft.irfft(freq_sols, num_grid_points)))
    labels = torch.stack((labels.real, labels.imag), dim=-1).view(labels.shape[0], -1)

    return labels
