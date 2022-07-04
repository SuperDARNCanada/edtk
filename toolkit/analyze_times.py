import os

import numpy as np
import re
import matplotlib.pyplot as plt
import argparse
import subprocess as sp

# This pattern will grab all numbers from a line
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


def execute_cmd(cmd):
    """
    Execute a shell command and return the output.

    Parameters
    ----------
    cmd: str
        The command

    Returns
    -------
    Decoded output of the command. String
    """
    output = sp.check_output(cmd, shell=True)
    return output.decode('utf-8')


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
    """Extract the last number per line from a file."""
    nums = []

    with open(filename, 'r') as f:
        for line in f:
            all_nums = rx.findall(line)
            if len(all_nums) > 0:
                nums.append(float(all_nums[-1]))

    return np.array(nums, np.float64)


def parse_file(data_dict: dict, filename: str):
    """
    Extracts only the lines from the file which meet the criteria of the data_dict.

    Parameters
    ----------
    data_dict: dict
        Dictionary containing information to aid in extracting the data
    filename: str
        Name of the file to search
    """
    # Get only the lines with the search phrase, and strip out all ANSI escape characters.
    tempfile = '/tmp/search.txt'
    search_cmd = 'grep "{}" {}'.format(data_dict['phrase'], filename) + \
                 ' | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" > ' + \
                 '{}'.format(tempfile)
    execute_cmd(search_cmd)

    # From the relevant lines, get the actual data
    extracted_nums = parse_nums(tempfile)
    return extracted_nums


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-filename", help="Name of the file to analyze.", required=True)
    parser.add_argument("-t", "--type", help="Type of data to extract. Supported types: 'num_sqns', 'dec_time', "
                                             "'parse_time', 'write_time', 'memcpy_time', 'bf_time', 'corr_time, "
                                             "'send_time', 'processing_time'. Parameter availability may depend "
                                             "Borealis version.", required=True)
    parser.add_argument("-V", "--version", help="Version of Borealis which generated the log file. Supported"
                                                "versions are 'v0.5' and 'v0.6'", default='v0.5')
    parser.add_argument("-d", "--plot_dir", help="Directory to save plot.")
    
    args = parser.parse_args()

    # This dictionary stores the relevant parameters for extracting each type of quantity from a log file.
    v05_data_type_params = {
        'num_sqns': {'bin_width': 4, 'phrase': 'Number of sequences', 'units': ''},
        'dec_time': {'bin_width': 10, 'phrase': 'Decimate time', 'units': 'ms'},
        'bf_time': {'bin_width': 2, 'phrase': 'Beamforming time', 'units': 'us'},
        'corr_time': {'bin_width': 10, 'phrase': 'ACF/XCF time', 'units': 'us'},
        'send_time': {'bin_width': 100, 'phrase': 'Fill + send', 'units': 'us'},
        'memcpy_time': {'bin_width': 5, 'phrase': 'Cuda memcpy time', 'units': 'ms'},
        'parse_time': {'bin_width': 1, 'phrase': 'Time to parse', 'units': 'ms'},
        'write_time': {'bin_width': 10, 'phrase': 'Time to write', 'units': 'ms'}
    }

    v06_data_type_params = {
        'num_sqns': {'bin_width': 4, 'phrase': 'Number of sequences', 'units': ''},
        'processing_time': {'bin_width': 1, 'phrase': 'Time to decimate, beamform and correlate', 'units': 'ms'},
        'send_time': {'bin_width': 0.1, 'phrase': 'Time to serialize and send', 'units': 'ms'},
        'memcpy_time': {'bin_width': 5, 'phrase': 'Time to copy samples', 'units': 'ms'},
        'parse_time': {'bin_width': 0.1, 'phrase': 'Time to parse', 'units': 'ms'},
        'write_time': {'bin_width': 10, 'phrase': 'Time to write', 'units': 'ms'}
    }

    if args.version == 'v0.5':
        version_dict = v05_data_type_params
    elif args.version == 'v0.6':
        version_dict = v06_data_type_params
    else:
        raise ValueError("Unknown Borealis version")

    if args.type not in version_dict.keys():
        raise KeyError("Parameter {} not supported in {}".format(args.type, args.version))
    else:
        param_dict = version_dict[args.type]

    if args.plot_dir is not None:
        plot_dir = args.plot_dir
    else:
        plot_dir, _ = os.path.split(args.input_filename)

    print(plot_dir)

    output_filename = os.path.basename(args.input_filename) + '-{}.png'.format(args.type)

    print(output_filename)

    data = parse_file(param_dict, args.input_filename)
    plot_samples(data, bin_size=param_dict['bin_width'], units=param_dict['units'], ylabel=param_dict['phrase'],
                 plot_dir=plot_dir, filename=output_filename)


