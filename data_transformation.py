import pandas as pd
import torch


def ten_to_df(tensor):

    numpy_array = tensor.detach().numpy()
    df = pd.DataFrame(numpy_array)

    return df


# Function to download a dataframe as an excel file
def df_to_excel(df, sheet_name=None):
    file_path = "C:/Users/yveje/Downloads/data.xlsx"
    df.to_excel(file_path, index=False)


# Function to normalize data


def normalize(tensor, tensor_min=None, tensor_max=None):
    if tensor_min is None and tensor_max is None:
        tensor_min = tensor.min(dim=0)[0]
        tensor_max = tensor.max(dim=0)[0]
    else:
        pass

    tensor_scaled = tensor.detach().clone()

    for i in range(tensor.size()[1]):
        if tensor_min[i] == 0 and tensor_max[i] == 0:
            continue
        if tensor_min[i] == tensor_max[i] and tensor_min[i] != 0:
            tensor_scaled[:, i] = 1
        else:
            tensor_scaled[:, i] = (tensor[:, i] - tensor_min[i]) / (tensor_max[i] - tensor_min[i])

    return [tensor_scaled, tensor_min, tensor_max]


# Function to scale back the normalized data

def denormalize(tensor_scaled, tensor_min, tensor_max):
    tensor = tensor_scaled.detach().clone()
    for i in range(tensor_scaled.size()[1]):
        tensor[:, i] = (tensor[:, i] * (tensor_max[i] - tensor_min[i])) + tensor_min[i]

    return tensor


# Function that converts a feature/label to its corresponding frequency solution.

def conv_to_freq(tensor):
    tensor = tensor.squeeze()
    freq_sol = torch.zeros((1, int((tensor.shape[0]) / 2)), dtype=torch.complex64)
    for i in range(freq_sol.shape[1]):
        freq_sol[0, i] = tensor[2 * i] + 1j * tensor[(2 * i + 1)]
    freq_sol = freq_sol.squeeze()
    return freq_sol
