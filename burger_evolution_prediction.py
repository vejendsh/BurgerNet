# Importing necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_generation import gen_freq_sols, gen_labels, gen_features, simulate, grid_points
from data_transformation import normalize, denormalize, conv_to_freq
from model import NeuralNetwork
import torch.fft as fft


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = 5000
test_size = int(train_size / 2)
dataset_size = int(train_size + test_size)


data_min = torch.full((102,), -100)
data_max = torch.full((102,), 100)
freq_sols = gen_freq_sols(dataset_size)
features = gen_features(freq_sols)
# features_data = df_to_excel(ten_to_df(normalize(features)[0]))
labels = gen_labels(freq_sols)
# labels_data = df_to_excel(ten_to_df(normalize(labels)[0]))
features_scaled = normalize(features, data_min, data_max)[0].to(device)
labels_scaled = normalize(labels, data_min, data_max)[0].to(device)

dataset = [(features_scaled[i], labels_scaled[i]) for i in range(len(labels))]
train_dataset = dataset[0:train_size]
test_dataset = dataset[train_size:]

batch_size = int(train_size / 10)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def run_epoch(dataloader, model, loss_fn, optimizer):

    counter = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        counter += 1

        if counter % 1 == 0:
            print_loss = loss.item()
            print(f"loss: {print_loss:>7f}")
            counter = 0

        # if batch % 100 == 0:


# def test_loop(dataloader, model, loss_fn):
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#
#     print(f"Test Error: \n Accuracy: {(100):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_model(epochs):
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 5e-4
    optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_2 = torch.optim.Adam(model.parameters(), lr=(learning_rate/5))
    optimizer_3 = torch.optim.Adam(model.parameters(), lr=(learning_rate/10))
    optimizer_4 = torch.optim.Adam(model.parameters(), lr=(learning_rate/20))

    for t in range(int((torch.floor(torch.tensor(epochs/4))).item())):
        run_epoch(train_dataloader, model, loss_fn, optimizer_1)
        if t % 1 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")

    for t in range(int((torch.floor(torch.tensor(epochs/4))).item())):
        run_epoch(train_dataloader, model, loss_fn, optimizer_2)
        # test_loop(test_dataloader)
        if t % 1 == 0:
            print(f"Epoch {epochs/4 + t + 1}\n-------------------------------")

    for t in range(int((torch.floor(torch.tensor(epochs/4))).item())):
        run_epoch(train_dataloader, model, loss_fn, optimizer_3)
        # test_loop(test_dataloader)
        if t % 1 == 0:
            print(f"Epoch {(2*epochs/4) + t + 1}\n-------------------------------")

    for t in range(int((torch.floor(torch.tensor(epochs/4))).item())):
        run_epoch(train_dataloader, model, loss_fn, optimizer_4)
        # test_loop(test_dataloader)
        if t % 1 == 0:
            print(f"Epoch {(3*epochs/4) + t + 1}\n-------------------------------")

    print("Done!")

    return model


num_epochs = 4000
trained_model = train_model(num_epochs)
torch.save(trained_model, 'model.pth')
