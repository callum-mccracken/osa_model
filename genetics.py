import model
import numpy as np
from random import random
import matplotlib.pyplot as plt

"""Module for Genetic Algorithm Code"""
generations = 10  # how many generations to run for
population_size = 100  # size of population in each generation
num_breeders = 10  # number that get to reproduce
num_offspring = (population_size - num_breeders) // num_breeders  # per breeder

class Sweep():
    def __init__(self, detuning, power):
        self.detuning = detuning
        self.power = power
        self.x, self.y = model.get_osa_output(detuning, power)
ideal_member = Sweep(model.ideal_detuning, model.ideal_power)


def generate_population():
    """Makes a population from scratch"""
    # currently just generating these randomly, should edit to be more useful
    # maybe use an evenly spaced grid between min/max laser params
    detunings = [2*random()-1 for _ in range(population_size)]
    powers = [2*random()-1 for _ in range(population_size)]
    return [Sweep(d, p) for d,p in zip(detunings, powers)]


y_ideal = ideal_member.y[ideal_member.y > model.noise_threshold]
def fitness(member:Sweep):
    # compare to ideal
    y_member = member.y[ideal_member.y > model.noise_threshold]
    return np.linalg.norm(y_ideal - y_member)


def get_fitness(pop):
    return np.array([fitness(member) for member in pop])


def mutate(breeder:Sweep):
    """for a given breeder, produce offspring"""
    old_detuning = breeder.detuning
    old_power = breeder.power
    offspring = []
    for i in range(num_offspring):
        d = old_detuning + 0.1*(2*random()-1)
        p = old_power + 0.1*(2*random()-1)
        offspring.append(Sweep(d, p))
    return offspring


def do_genetics(plot_path=True):
    pop = generate_population()
    best_d = []
    best_p = []
    for n in range(generations):
        print('generation', n)
        fitnesses = get_fitness(pop)
        highest_indices = np.argsort(fitnesses)[:num_breeders]
        best_d.append(pop[highest_indices[0]].detuning)
        best_p.append(pop[highest_indices[0]].power)
        breeders = [pop[i] for i in highest_indices]
        for b in breeders:
            pop += mutate(b)  # add offspring
    fitnesses = get_fitness(pop)
    highest_index = np.argsort(fitnesses)[0]
    best_member = pop[highest_index]
    print('best member:', best_member.detuning, best_member.power)

    if plot_path:
        check_fitness(show=False)
        plt.scatter(best_d, best_p)
        plt.plot([best_member.detuning], [best_member.power], 'r', marker='x', markersize=12)
        plt.show()
    return best_member


def check_fitness(show=True):
    # to check if fitness as defined above is a good thing to try to optimize
    n_d, n_p = (10, 10)
    ds = np.linspace(model.ideal_detuning-1, model.ideal_detuning+1, n_d)
    ps = np.linspace(model.ideal_power-1, model.ideal_power+1, n_p)
    d, p = np.meshgrid(ds, ps)
    f = np.empty_like(d)
    for i, di in np.ndenumerate(d):
        #print(i)
        pi = p[i]
        member = Sweep(di, pi)
        f[i] = fitness(member)
    plt.pcolormesh(d, p, f, shading='auto')
    plt.colorbar()
    if show:
        plt.show()


if __name__ == "__main__":
    best_member = do_genetics()
