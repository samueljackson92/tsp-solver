import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness(simulator):
    fig, ax = plt.subplots()

    ax.plot(simulator.get_averge_fitness(), label='Average Fitness')
    ax.plot(simulator.get_min_fitness(), label='Min Fitness')
    ax.plot(simulator.get_max_fitness(), label='Max Fitness')

    ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Fitness')

    plt.show()


def plot_solution(data, solution):
    plt.scatter(data[:, 0], data[:, 1])

    for a, b in zip(solution, np.roll(solution, -1)):
        m = np.vstack((data[a], data[b]))
        plt.plot(m[:, 0], m[:, 1], color='red')

    plt.show()
