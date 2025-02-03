import numpy as np
import matplotlib.pyplot as plt

def init_lattice(N):
    """Initialize a random N x N lattice with spins +1 or -1."""
    return np.random.choice([-1, 1], size=(N, N))

def metropolis_step(lattice, J, T):
    """Perform one Metropolis-Hastings step on the lattice."""
    N = lattice.shape[0]
    # Randomly select a site
    i, j = np.random.randint(0, N, size=2)
    s = lattice[i, j]

    # Calculate sum of neighboring spins with periodic boundary conditions
    neighbors = (
        lattice[(i+1) % N, j] +
        lattice[(i-1) % N, j] +
        lattice[i, (j+1) % N] +
        lattice[i, (j-1) % N]
    )

    delta_E = 2 * J * s * neighbors

    # Acceptance condition
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
        lattice[i, j] *= -1  # Flip the spin

    return lattice

def run_ising_model(N=50, J=1.0, T=2.0, steps=1000):
    """Run the Metropolis-Hastings algorithm for the Ising model."""
    lattice = init_lattice(N)
    for _ in range(steps):
        metropolis_step(lattice, J, T)
    return lattice

def plot_lattice(lattice):
    """Display the lattice configuration using matplotlib."""
    plt.imshow(lattice, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

# some utilities
def critical_temperature(J):
    """Compute the critical temperature for the 2D Ising model."""
    return (2 * J) / np.log(1 + np.sqrt(2))

# Example usage
if __name__ == '__main__':
    # Parameters
    N = 50  # Lattice size
    J = 1.0  # Interaction strength (ferromagnetic if positive)
    T = 2.0  # Temperature
    steps = 10000  # Number of Metropolis steps

    # Run simulation
    lattice = run_ising_model(N=N, J=J, T=T, steps=steps)

    # Plot the final configuration
    plot_lattice(lattice)
