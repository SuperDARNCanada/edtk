"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (Rohde & Schwarz Data Plotting)

Author: Adam Lozinsky
Date: October 6, 2021
Affiliation: University of Saskatchewan

Typically SuperDARN engineers will make a series of measurements for each antennas RF path using a
Rhode & Schwarz ZVH or similar spectrum analyzer. These measurements can be converted into .csv files.
The files contain different data based on the instrument settings, but it is per antenna. It is preferred
to plot all the data for each antenna on one plot so differences and outliers are easily visible. This tool
will produce those common plots from the .csv files.

Use 'python zvh_tools.py --help' to discover options if running directly from command line.
"""

from dataclasses import dataclass, field
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True, order=True)
class RSData:
    """Data class for keeping the Rohde & Schwarz ZVH data."""
    name: str
    freq: float
    vswr: float
    magnitude: float
    phase: float


@dataclass(order=True)
class RSAllData:
    """Data class holding all loaded Rohde & Schwarz ZVH data for a requested job."""
    site: str = field(default_factory=str)
    date: str = field(default_factory=str)
    names: list = field(default_factory=list)
    datas: list = field(default_factory=list)


def read_data(directory, pattern, verbose=False, site=''):
    """
    Load the Rohde & Schwarz data from .csv files from either a parent directory given a file pattern or from a
    directory directly. The data is then loaded into a dataclass and returned for further processing.

    Parameters
    ----------
        directory : str
            The directory or parent directory containing the .csv files.
        pattern : str
            The file naming pattern of files to load; eg. rkn_vswr would yield all rkn_vswr*.csv in directory tree.
        verbose : bool
            True will print more information about whats going on, False squelches.
        site : str
            Name of the site the data was taken from; used in naming plots and plot titles.

    Returns
    -------
        all_data : dataclass
            A dataclass containing all the data for each antenna from the Rohde & Schwarz .csv files.
    """

    files = glob.glob(directory + '/*/' + pattern + '*.csv')
    if files == []:
        files = glob.glob(directory + pattern + '*.csv')
    verbose and print("files found:\n", files)

    all_data = RSAllData()
    all_data.site = site
    for file in files:
        name = os.path.basename(file).replace('.csv', '')
        verbose and print(f'loading file: {file}')
        df = pd.read_csv(file, encoding='cp1252')
        skiprows = 0
        for index, row in df.iterrows():
            skiprows += 1
            if 'date' in str(row).lower():
                date = row[1].replace(' ', '').split('/')
                date = '-'.join(date[::-1])
                all_data.date = date
            if '[hz]' in str(row).lower():
                break

        # The ZVH .csv files are in format cp1252 not utf-8 so using utf-8 will break on degrees symbol.
        df = pd.read_csv(file, skiprows=skiprows, encoding='cp1252')
        keys = list(df.keys())
        freq = None
        vswr = None
        magnitude = None
        phase = None
        for key in keys:
            if 'unnamed' in key.lower():  # Break from loop after the first ZVH sweep.
                verbose and print('\t-end of first sweep')
                break
            if 'freq' in key.lower():
                freq = df[key]
                verbose and print(f'\t-FREQUENCY data found in: {name}')
            if 'vswr' in key.lower():
                vswr = df[key]
                verbose and print(f'\t-VSWR data found in: {name}')
            if 'mag' in key.lower():
                magnitude = df[key]
                verbose and print(f'\t-MAGNITUDE data found in: {name}')
            if 'pha' in key.lower():
                phase = df[key]
                verbose and print(f'\t-PHASE data found in: {name}')

        data = RSData(name=name, freq=freq, vswr=vswr, magnitude=magnitude, phase=phase)
        all_data.names.append(name)
        all_data.datas.append(data)

    return all_data


def plot_rx_path(data, directory=''):
    """
    Create a plot of frequency vs. magnitude and frequency vs. phase for each antenna receive path.

    Parameters
    ----------
        data : dataclass
            A dataclass containing Rohde & Schwarz ZVH measured data; must contain vswr and frequency.
        directory : str
            The output file directory to save the plot in.

    Returns
    -------
        None
    """

    # Pretty plot configuration.
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    outfile = f'{directory}rx_path_{data.site}_{data.date}.png'
    mean_magnitude = 0.0
    mean_phase = 0.0
    total_antennas = 0.0
    xmin = 20.0e6
    xmax = 0.0

    fig, ax = plt.subplots(2, 1, figsize=[13, 8])
    fig.suptitle(f'Rohde & Schwarz Data: RX Path per Antenna\n{data.site} {data.date}')
    for index, name in enumerate(data.names):
        mean_magnitude += data.datas[index].magnitude
        mean_phase += data.datas[index].phase
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        ax[0].plot(data.datas[index].freq, data.datas[index].magnitude, label=data.datas[index].name)
        ax[1].plot(data.datas[index].freq, data.datas[index].phase, label=data.datas[index].name)

    mean_magnitude /= total_antennas
    mean_phase /= total_antennas
    ax[0].plot(data.datas[0].freq, mean_magnitude, '--k', label='mean magnitude')
    ax[1].plot(data.datas[0].freq, mean_phase, '--k', label='mean phase')

    plt.legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim([xmin, xmax])
    ax[1].set_xlim([xmin, xmax])
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax[1].set_xlabel('Frequency [MHz]')
    ax[0].set_ylabel('Magnitude [dBm]')
    ax[1].set_ylabel('Phase [°]')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'rx path file created at: {outfile}')
    return


def plot_vswr(data, directory=''):
    """
    Create a plot of frequency vs. voltage standing wave ratio (vswr) for each antenna.

    Parameters
    ----------
        data : dataclass
            A dataclass containing Rohde & Schwarz ZVH measured data; must contain vswr and frequency.
        directory : str
            The output file directory to save the plot in.

    Returns
    -------
        None
    """

    # Pretty plot configuration.
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    outfile = f'{directory}vswr_{data.site}_{data.date}.png'
    mean_vswr = 0.0
    total_antennas = 0.0
    xmin = 20.0e6
    xmax = 0.0

    plt.figure(figsize=[13, 8])
    plt.suptitle(f'Rohde & Schwarz Data: VSWR per Antenna\n{data.site} {data.date}')
    for index, name in enumerate(data.names):
        mean_vswr += data.datas[index].vswr
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        plt.plot(data.datas[index].freq, data.datas[index].vswr, label=data.datas[index].name)

    mean_vswr /= total_antennas
    plt.plot(data.datas[0].freq, mean_vswr, '--k', label='mean')

    # plt.legend(loc='best', fancybox=True, ncol=3)
    plt.legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.2])
    plt.grid()
    plt.xlim([xmin, xmax])
    plt.ylim([1.0, 3.0])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('VSWR')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'vswr plot created at: {outfile}')
    return


def main():
    parser = argparse.ArgumentParser(description='SuperDARN Canada© -- Engineering Diagnostic Tools Kit: '
                                                 '(Rohde & Schwarz Data Plotting) '
                                                 'Given a set of Rohde & Schwarz ZVH files that have been converted to'
                                                 '.csv format this program will generate a series of comparison plots'
                                                 'for engineering diagnostics.')
    parser.add_argument('-s', '--site', type=str, help='name of the site this data is from, eg: Inuvik, Saskatoon,...')
    parser.add_argument('-d', '--directory', type=str, help='directory containing ZVH files with data to be plotted.')
    parser.add_argument('-o', '--outdir', type=str, default='', help='directory to save output plots.')
    parser.add_argument('-p', '--pattern', type=str, help='the file naming pattern less the appending numbers.')
    parser.add_argument('-v', '--verbose', action='store_true', help='explain what is being done verbosely.')
    parser.add_argument('-m', '--mode', type=str, help='select the plot mode, eg: vswr or path.')
    args = parser.parse_args()
    directory = args.directory
    outdir = args.outdir
    if outdir == '':
        outdir = directory
    pattern = args.pattern

    if args.directory is None:
        directory = ''
    if args.pattern is None:
        pattern = ''

    data = read_data(directory, pattern, args.verbose, args.site)

    if args.mode == 'vswr':
        plot_vswr(data, directory=outdir)
    elif args.mode == 'path':
        plot_rx_path(data, directory=outdir)
    else:
        print('Select a mode: vswr or path')

    return None


if __name__ == '__main__':
    main()
