import pennylane as qml
from pennylane import numpy as np

# While PennyLane provides a basic qubit simulator ('default.qubit') and a basic CV Gaussian simulator ('default.gaussian'), the true power of PennyLane comes from its plugin ecosystem, allowing quantum computations to be run on a variety of quantum simulator and hardware devices.

# For this circuit, we will be using the 'strawberryfields.fock' device to construct a QNode. This allows the underlying quantum computation to be performed using the Strawberry Fields Fock backend.

# As usual, we begin by importing PennyLane and the wrapped version of NumPy provided by PennyLane:

dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=2)

@qml.qnode(dev_fock, diff_method="parameter-shift")
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))

# Let’s now use one of the built-in PennyLane optimizers in order to carry out photon redirection. Since we wish to maximize the mean photon number of the second wire, we can define our cost function to minimize the negative of the circuit output.

def cost(params):
    return -photon_redirection(params)

init_params = np.array([0.01, 0.01])
print(cost(init_params))

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))
