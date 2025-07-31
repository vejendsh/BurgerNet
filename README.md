# Burger_pred
Burger_pred is a neural network that can predict the transient solution of 1D periodic Burger's Equation for any arbitrary initital condition and viscosity. It uses fourier transforms to condense the complex time domain features into simple frequency domain features for efficient processing. This framework works with any arbitrary linear or non-linear time-dependent PDE but the current project implements it on the 1D periodic Burger's Equation.

Files dedicated to data generation, pre-processing, and model testing are also present. Instructions to run this code locally are provided below.

1. Python 3.0 is required to utilize the code in this repository.
2. Download all the files present in the repository and save them in a separate folder.
3. Open any Command Line Interface (CLI) or Terminal.
4. Navigate to the folder where the files have been installed.
5. Run the command: "python .\burger_evolution_prediction.py" (exclude double quotes) to train the model.
6. Run the command: "python .\testing_nn.py" (exclude double quotes) to see how the model performs on a randomly generated testing sample.
7. Repeat step 6 to check the model performance for different testing examples.

Browse the files with names starting with "Demo" to see a demonstration of model performance for arbitrary initial solutions.
