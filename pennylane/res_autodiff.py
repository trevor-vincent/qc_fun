from jax import numpy as jnp
from jax import grad, jit
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def f(x):
    return (x-2.0)**2

f_prime = jit(grad(f)) # use jit(grad(f)) if you want the faster, compiled version of grad(f)

f_values = []
f_prime_values = []
x_values = []
lr = 0.01
x = 3.0
for step in tqdm(range(1000)):
    x = x - lr*f_prime(x)
    x_values.append(x)
    f_values.append(f(x)) # we should be able to get f(x) from f_prime without recomputing it here...
    f_prime_values.append(f_prime(f(x)))
    
plt.plot(x_values)
plt.plot(f_values)
plt.plot(f_prime_values)
plt.savefig("plot.png")
