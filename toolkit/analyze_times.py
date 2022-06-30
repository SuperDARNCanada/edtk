import numpy as np
import re
import matplotlib.pyplot as plt
import argparse


numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


def calculate_statistics(samples: np.array):
    """Calculate mean, min, max, and standard deviation of an array."""
    N = samples.size
    mean = np.mean(samples)
    mean2 = np.mean(samples * samples)
    stddev = np.sqrt(mean2 - mean*mean)
    min_val = np.min(samples)
    max_val = np.max(samples)

    return mean, stddev, min_val, max_val, N


def plot_samples(samples: np.array, bin_size: int = 10, units: str = '', ylabel: str = '', plot_dir: str = '.',
                 filename: str = ''):
    """
    Calculate statistics about the samples, print them, and plot.

    Parameters
    ----------
        samples: np.array
            Array of samples of a given quantity (number of sequences, processing time, etc.)
        bin_size: int
            Width of bins for plotting histogram.
        units: str
            Units for the given parameter.
        ylabel: str
            Descriptor of the ordinate (y-axis) values.
        plot_dir: str
            Directory to save plot in. Default is the current directory.
        filename: str
            Name to give the plot.
    """
    mean, stddev, min_val, max_val, N = calculate_statistics(samples)

    print('N: {}'.format(N))
    print("Mean: {}".format(mean))
    print("Std. Dev: {}".format(stddev))
    print('Max: {}'.format(max_val))
    print('Min: {}'.format(min_val))

    bins = np.arange(np.floor(min_val), np.ceil(max_val) + bin_size, bin_size)
    print(bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.hist(samples, bins)
    ax1.set_xlabel(f'{ylabel} ({units})')
    ax1.set_ylabel('Counts')
    ax1.set_title('Histogram')

    ax2.plot(samples)
    ax2.set_xlabel('Sample number')
    ax2.set_ylabel(f'{ylabel} ({units})')
    ax2.set_title('Time series')

    fields = filename.split('.')
    plot_name = '.'.join(fields[:-1])
    plt.savefig(f'{plot_dir}/{plot_name}.png', bbox_inches='tight')


def parse_nums(filename: str):
    """Extract the first number per line from a file."""
    nums = []

    with open(filename, 'r') as f:
        for line in f:
            all_floats = rx.findall(line)
            if len(all_floats) > 0:
                nums.append(float(all_floats[0]))

    return np.array(nums, np.float64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Name of the file to analyze.", required=True)
    parser.add_argument("-t", "--type", help="Type of data to extract. Supported types: 'num_sqns', 'dec_time', "
                                             "'parse_time', 'write_time', 'memcpy_time', 'bf_time', 'corr_time, "
                                             "'send_time'.", required=True)
    parser.add_argument("-d", "--plot_dir", help="Directory to save plot.", default='')
    
    args = parser.parse_args()

    # TODO(Remington): Determine how to parse file based on type parameter

