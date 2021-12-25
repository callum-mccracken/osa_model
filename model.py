import matplotlib.pyplot as plt
import numpy as np
import random

# axis limits for graph
x_min = -10
x_max = 10
n_points = 1000
x = np.linspace(x_min, x_max, n_points)

# parameters for peaks
peak_separation = 6
peak_decay = 5  # controls width of peaks
wide_decay = 0.001  # controls width of overall envelope of peaks

# peaks / values below this threshold will have noise added
noise_threshold = 0.1

# parameters for the sweet spot of the model, that the algorithm should learn
ideal_detuning = 1
ideal_power = 1
detuning_decay = 1  # sweet spot defined by Gaussians, decay rates for those
power_decay = 1

# generate peak positions
first_peak_x = x_min + random.random() * peak_separation
n_peaks = int((x_max - x_min) / peak_separation)
peak_xs = [first_peak_x + n * peak_separation for n in range(-1, n_peaks + 1)]
middle_peak = peak_xs[int(len(peak_xs)/2)]

def comb(x, ps=peak_separation, pd=peak_decay, wd=wide_decay):
    """the function that looks like a comb (evenly spaced Gaussians)"""
    # add a bunch of Gaussians, one per peak
    y = sum([np.exp(-pd*(x - x_i)**2) for x_i in peak_xs])
    # multiply by an overall Gaussian envelope
    y *= np.exp(-wd*(x - middle_peak)**2)
    return y


def add_noise(y, noise_threshold=noise_threshold):
    """takes y and adds noise to all values below the noise threshold"""
    y_to_make_noisy = np.where(y < noise_threshold)
    noise = np.array([random.random() for _ in x])
    y[y_to_make_noisy] *= noise[y_to_make_noisy]
    return y



def get_osa_output(detuning, power):
    """given detuning and power (numbers), return x, y = osa output (arrays)"""
    # define gaussian filters
    detuning_filter = np.exp(-detuning_decay*(detuning - ideal_detuning)**2)
    power_filter = np.exp(-power_decay*(power - ideal_power)**2)
    # generate a comb, apply filters, and add noise
    y = add_noise(comb(x) * detuning_filter * power_filter)
    return x, y


def get_osa_outputs(detunings, powers):
    xs = np.array([x]*len(detunings))
    ys = np.empty_like(xs)
    for i in range(len(detunings)):
        d, p = detunings[i], powers[i]
        xs[i], ys[i] = get_osa_output(d, p)
    return xs, ys


if __name__ == "__main__":
    xs, ys = get_osa_outputs(np.array([2,3,4]),np.array([2,3,4]))
    for xi, yi, ci in zip(xs, ys, ['r', 'g', 'b']):
        plt.plot(xi, yi, ci)
    plt.plot(*get_osa_output(1,1), 'k')
    plt.show()
