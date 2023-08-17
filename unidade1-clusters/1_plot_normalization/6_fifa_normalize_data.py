import matplotlib.pyplot as plt
from src.utils import load_fifa_dataset
from scipy.cluster.vq import whiten

fifa = load_fifa_dataset()

# Scale wage and value
fifa['scaled_wage'] = ____(fifa[____])
fifa['scaled_value'] = ____(fifa[____])


# Plot the two columns in a scatter plot
fifa.__(x=__, y=__, kind='scatter')
plt.__()