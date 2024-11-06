"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (Vector Network Analyzer Data Plotting)

Author: Adam Lozinsky
Date: October 6, 2021
Affiliation: University of Saskatchewan

Typically SuperDARN engineers will make a series of measurements for each antennas RF path using a
Rohde & Schwarz ZVH or similar vector network analyzer (I.E. Copper Mountain TR VNA). These
measurements can be converted into .csv files. The files contain different data based on the
instrument settings, but it is per antenna. It is preferred to plot all the data for each antenna on
one plot so differences and outliers are easily visible. This tool will produce those common plots
from the .csv files.

Use 'python zvh_tools.py --help' to discover options if running directly from command line.
"""

from dataclasses import dataclass, field
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


@dataclass(frozen=True, order=True)
class RSData:
    """Data class for keeping the VNA data."""
    name: str
    freq: float
    vswr: float
    magnitude: float
    phase: float


@dataclass(order=True)
class RSAllData:
    """Data class holding all loaded VNA data for a requested job."""
    site: str = field(default_factory=str)
    date: str = field(default_factory=str)
    vna: str = field(default_factory=str)
    names: list = field(default_factory=list)
    datas: list = field(default_factory=list)


def read_data(directory, pattern, date, verbose=False, site='', vna='zvh'):
    """
    Load the VNA data from .csv files from either a parent directory given a file pattern or from a
    directory directly. The data is then loaded into a dataclass and returned for further processing.
    
    For Rohde & Schwarz ZVH, data must be read in from the specific R&S format.
    For Copper Mountain VNA, data can be read in directly from utf-8 CSV files

    Parameters
    ----------
        directory : str
            The directory or parent directory containing the .csv files.
        pattern : str
            The file naming pattern of files to load; eg. rkn_vswr would yield all rkn_vswr*.csv in directory tree.
        date : str
            Recording date of data to be plotted in the form yyyy-mm-dd
        verbose : bool
            True will print more information about whats going on, False squelches.
        site : str
            Name of the site the data was taken from; used in naming plots and plot titles.
        vna : str
            Vector network analyzer that produced the data to be plotted. Options are 'zvh' for 
            the Rohde & Schwarz ZVH, or 'trvna' for the Copper Mountain TR VNA

    Returns
    -------
        all_data : dataclass
            A dataclass containing all the data for each antenna from the Rohde & Schwarz .csv files.
    """

    files = glob.glob(directory + '/*/' + pattern + '*.csv')
    if files == []:
        files = sorted(glob.glob(directory + pattern + '*.csv'))
    verbose and print("files found:\n", files)

    all_data = RSAllData()
    all_data.site = site
    all_data.vna = vna
    for file in files:
        name = os.path.basename(file).replace('.csv', '')
        verbose and print(f'loading file: {file}')
        if vna == 'zvh':
            df = pd.read_csv(file, encoding='cp1252')
            all_data.date = date
            skiprows = 0
            for index, row in df.iterrows():
                skiprows += 1
                if 'date' in str(row).lower():
                    date = row.iloc[1].replace(' ', '').split('/')
                    date = '-'.join(date[::-1])
                    date = f'{datetime.strptime(date, "%Y-%m-%d").date()}'
                    all_data.date = date
                if '[hz]' in str(row).lower():
                    break

            # The ZVH .csv files are in format cp1252 not utf-8 so using utf-8 will break on degrees symbol.
            df = pd.read_csv(file, skiprows=skiprows, encoding='cp1252')
        # trvna only stores 5 columns of values in csv format (no headings)
        # Frequency |  Re(VSWR) |  Im(VSWR) | Re(Phase) | Im(Phase)
        # OR
        # Frequency |  Re(Mag)  |  Im(Mag)  | Re(Phase) | Im(Phase)
        # Either way we need columns 0, 1 and 3 (don't need imaginary parts)
        elif vna == 'trvna':
            df_temp = pd.read_csv(file)
            df = {}
            key_array = []
            for key in df_temp.keys():
                key_array.append(key)
            if 'vswr' in pattern.lower():
                df['Frequency [Hz]'] = df_temp[key_array[0]]
                df['VSWR'] = df_temp[key_array[1]]
                df['Phase'] = df_temp[key_array[3]]
            if 'rxpath' in pattern.lower():
                df['Frequency [Hz]'] = df_temp[key_array[0]]
                df['Magnitude'] = df_temp[key_array[1]]
                df['Phase'] = df_temp[key_array[3]]
            all_data.date = date
        else:
            print(f'Unknown VNA {vna}: Exiting...')
            exit(1)
        
        keys = list(df.keys())
        freq = None
        vswr = None
        magnitude = None
        phase = None
        for key in keys:
            # if 'unnamed' in key.lower():  # Break from loop after the first ZVH sweep.
            #     verbose and print('\t-end of first sweep')
            #     break
            if 'freq' in key.lower():
                freq = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-FREQUENCY data found in: {name}')
            if 'vswr' in key.lower():
                vswr = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-VSWR data found in: {name}')
            if 'mag' in key.lower():
                magnitude = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-MAGNITUDE data found in: {name}')
            if 'pha' in key.lower():
                phase = pd.to_numeric(df[key], errors='coerce')
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

    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_COLOURS = 10

    outfile = f'{directory}rx_path_{data.site.lower()}_{data.date}.png'
    mean_magnitude = 0.0
    mean_phase = 0.0
    total_antennas = 0.0
    xmin = 8.0e6
    xmax = 20.0e6

    fig, ax = plt.subplots(2, 1, figsize=[13, 8])
    fig.suptitle(f'{data.vna.upper()} Data: RX Path per Antenna\n{data.site.upper()} {data.date}')
    for index, name in enumerate(data.names):
        mean_magnitude += data.datas[index].magnitude
        mean_phase += data.datas[index].phase
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        ax[0].plot(data.datas[index].freq/1E+6, data.datas[index].magnitude, label=data.datas[index].name,
                    linestyle=LINE_STYLES[int(index/NUM_COLOURS)])
        ax[1].plot(data.datas[index].freq/1E+6, data.datas[index].phase, label=data.datas[index].name)

    #mean_magnitude /= total_antennas
    #mean_phase /= total_antennas
    #ax[0].plot(data.datas[0].freq/1E+6, mean_magnitude, '--k', label='mean')
    #ax[1].plot(data.datas[0].freq/1E+6, mean_phase, '--k', label='mean')

    xmin /= 1E+6
    xmax /= 1E+6

    ax[0].legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim([xmin, xmax])
    # ax[0].set_ylim([25,30])
    ax[1].set_xlim([xmin, xmax])
    # ax[0].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    # ax[1].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax[1].set_xlabel('Frequency [MHz]')
    ax[0].set_ylabel('Magnitude [dB]')
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
    
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_COLOURS = 10

    outfile = f'{directory}vswr_{data.site.lower()}_{data.date}.png'
    mean_vswr = 0.0
    total_antennas = 0.0
    xmin = 8.0e6
    xmax = 20.0e6

    plt.figure(figsize=[13, 8])
    plt.suptitle(f'{data.vna.upper()} Data: VSWR per Antenna\n{data.site.upper()} {data.date}')
    for index, name in enumerate(data.names):
        mean_vswr += data.datas[index].vswr
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        plt.plot(data.datas[index].freq/1E+6,
                 data.datas[index].vswr,
                 label=data.datas[index].name,
                 linestyle=LINE_STYLES[int(index/NUM_COLOURS)])

    mean_vswr /= total_antennas
    plt.plot(data.datas[0].freq/1E+6, mean_vswr, '--k', label='mean')

    xmin /= 1E+6
    xmax /= 1E+6

    # plt.legend(loc='best', fancybox=True, ncol=3)
    plt.legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.2])
    plt.grid()
    plt.xlim([xmin, xmax])
    plt.ylim([1.0, 3.0])
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('VSWR')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'vswr plot created at: {outfile}')
    return


def plot_sky_noise(data, directory=''):
    """
    Create a plot of frequency vs. magnitude antenna test.

    Parameters
    ----------
        data : dataclass
            A dataclass containing Rohde & Schwarz ZVH measured data; must contain magnitude and frequency.
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

    outfile = f'{directory}sky_noise_{data.site}_{data.date}.png'
    mean_magnitude = 0.0
    total_antennas = 0.0
    xmin = 20.0e6
    xmax = 0.0

    fig, ax = plt.subplots(1, 1, figsize=[13, 8])
    fig.suptitle(f'Rohde & Schwarz Data: Sky Noise\n{data.site} {data.date}')
    for index, name in enumerate(data.names):
        #mean_magnitude += data.datas[index].magnitude
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        ax.plot(data.datas[index].freq, data.datas[index].magnitude, label=data.datas[index].name)

    #mean_magnitude /= total_antennas
    #ax.plot(data.datas[0].freq, mean_magnitude, '--k', label='mean magnitude')

    plt.legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax.grid()
    ax.set_xlim([xmin, xmax])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Magnitude [dBm]')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'sky noise file created at: {outfile}')
    return


def main():
    parser = argparse.ArgumentParser(description='SuperDARN Canada© -- Engineering Diagnostic Tools Kit: '
                                                 '(Rohde & Schwarz Data Plotting) '
                                                 'Given a set of Rohde & Schwarz ZVH files that have been converted to'
                                                 '.csv format this program will generate a series of comparison plots'
                                                 'for engineering diagnostics.')
    parser.add_argument('-s', '--site', type=str, help='name of the site this data is from, eg: INV, SAS,...')
    parser.add_argument('-d', '--directory', type=str, help='directory containing ZVH files with data to be plotted.')
    parser.add_argument('-o', '--outdir', type=str, default='', help='directory to save output plots.')
    parser.add_argument('-p', '--pattern', type=str, help='the file naming pattern less the appending numbers.')
    parser.add_argument('-v', '--verbose', action='store_true', help='explain what is being done verbosely.')
    parser.add_argument('-m', '--mode', type=str, help='select the plot mode, options(vswr, path, sky).')
    parser.add_argument('--vna', type=str, help='select VNA to plot for, options(zvh, trvna).')
    parser.add_argument('--date', type=str, help='date of the data to be plotted (yyyy-mm-dd)')
    args = parser.parse_args()
    directory = args.directory
    outdir = args.outdir
    if outdir == '':
        outdir = directory
    pattern = args.pattern
    date = args.date

    if args.directory is None:
        directory = ''
    if args.pattern is None:
        pattern = ''
    if args.vna is None:
        vna = 'zvh'
    else:
        vna = args.vna
    if args.date is None:
        date = ''

    data = read_data(directory, pattern, date, args.verbose, args.site, vna)

    if args.mode == 'vswr':
        plot_vswr(data, directory=outdir)
    elif args.mode == 'path':
        plot_rx_path(data, directory=outdir)
    elif args.mode == "sky":
        plot_sky_noise(data, directory=outdir)
    else:
        print('Select a mode: vswr, path, or sky')

    return None


if __name__ == '__main__':
    main()
    # data = read_data('/home/radar/testing/vna_plotting/', '', False, 'rkn', 'trvna')
