from data_generation import gen_freq_sols, simulate, gen_features, grid_points
from data_transformation import normalize, denormalize, conv_to_freq
import torch.fft as fft
import matplotlib.pyplot as plt
from model import NeuralNetwork
import torch

data_min = torch.full((102,), -100)
data_max = torch.full((102,), 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

# Testing model prediction for n time steps
# num_time_steps = 1
test_freq_sol = gen_freq_sols(1)
test_time_sol = fft.irfft(test_freq_sol, 101)
test_updated_time_sol = simulate(test_time_sol)
test_feature = gen_features(test_freq_sol)
test_feature = normalize(test_feature, data_min, data_max)[0].to(device)
old_prediction = denormalize(model(test_feature), data_min, data_max).to(device)

# old_prediction = torch.zeros_like(test_feature)
# for i in range(num_time_steps):
#     old_prediction = model(test_feature)
#     test_feature = old_prediction
#
# old_prediction = denormalize(old_prediction, data_min, data_max)

pred_freq_sol = conv_to_freq(old_prediction)
pred_time_sol = fft.irfft(pred_freq_sol, 101)


plt.plot(grid_points, test_time_sol.t().detach().cpu().numpy(), color="y", label="initial")
plt.plot(grid_points, test_updated_time_sol.t().detach().cpu().numpy(), color="b", label="updated")
plt.plot(grid_points, pred_time_sol.t().detach().cpu().numpy(), color="r", label="prediction")

plt.grid()
plt.legend()
plt.show()
