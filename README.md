# TimeNetPDE
TimeNetPDE is a neural network that can predict the solution of a time-dependent partial differential equation (PDE) at any time given its initial solution. It can work with both linear and non-linear PDEs. The current implementation uses the 1D Burger's Equation but the same framework can be used for any non-linear time-dependent PDE. Files dedicated to data generation, pre-processing, and model testing are also present. Instructions to run this code locally are provided below.

1. Python 3.0 is required to utilize the code in this repository.
2. Download all the files present in the repository and save them in a separate folder.
3. Open any Command Line Interface (CLI) or Terminal.
4. Navigate to the folder where the files have been installed.
5. Run the command: "python .\burger_evolution_prediction.py" (exclude double quotes) to train the model.
6. Run the command: "python .\testing_nn.py" (exclude double quotes) to see how the model performs on a randomly generated testing sample.
7. Repeat step 6 to check the model performance for different testing examples.

Browse the files with names starting with "Demo" to see a demonstration of model performance for arbitrary initial solutions.
