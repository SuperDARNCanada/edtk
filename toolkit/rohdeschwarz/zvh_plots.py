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
    phase_unwrapped: float


@dataclass(order=True)
class RSAllData:
    """Data class holding all loaded VNA data for a requested job."""
    site_code: str = field(default_factory=str)
    site_name: str = field(default_factory=str)
    date: str = field(default_factory=str)
    vna: str = field(default_factory=str)
    names: list = field(default_factory=list)
    datas: list = field(default_factory=list)


def read_data(directory, pattern, date, verbose, site_code, site_name, vna, mode):
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
        mode : str
            The type of data collected. Options are vswr, rxpath, skynoise

    Returns
    -------
        all_data : dataclass
            A dataclass containing all the data for each antenna from the Rohde & Schwarz .csv files.
    """

    files = glob.glob(f"{directory}/*/{pattern}*.csv")
    if files == []:
        files = sorted(glob.glob(f"{directory}/{pattern}*.csv"))
    verbose and print("files found:\n", files)

    all_data = RSAllData()
    all_data.site_code = site_code
    all_data.site_name = site_name
    all_data.vna = vna
    all_data.date = date

    for file in files:
        name = os.path.basename(file).replace('.csv', '')
        verbose and print(f'loading file: {file}')
        if vna == 'zvh4':
            # Determine which row the data starts on
            df = pd.read_csv(file, encoding='cp1252')
            skiprows = 0
            endrow = 0
            for index, row in df.iterrows():
                skiprows += 1
                # ZVH4 records specific time data was collected. Use this for the date instead.
                if 'date' in str(row).lower():
                    date = datetime.strptime(row.iloc[1], "%m/%d/%Y")
                    iso_date = datetime.strftime(date, "%Y-%m-%d")
                    all_data.date = iso_date
                if '[hz]' in str(row).lower():
                    break

            # Determine what row the data ends on
            for index, row in df.iterrows():
                endrow += 1
                if 'memory' in str(row).lower():
                    endrow -=1 # Move back one row to account for empty line
                    break

            # The ZVH .csv files are in format cp1252 not utf-8 so using utf-8 will break on degrees symbol.
            df = pd.read_csv(file, skiprows=skiprows, nrows=endrow-skiprows, encoding='cp1252')

        elif vna == 'trvna':
            # trvna only stores 5 columns of values in csv format (no headings)
            # Frequency |  Re(VSWR) |  Im(VSWR) | Re(Phase) | Im(Phase)    -- for vswr
            # OR
            # Frequency |  Re(Mag)  |  Im(Mag)  | Re(Phase) | Im(Phase)    -- for rxpath
            # Either way we need columns 0, 1 and 3 (don't need imaginary parts)
            df_temp = pd.read_csv(file)

            # Check if headers have been manually added to the csv file
            first_row = df_temp.head(0)
            if 'freq' in str(first_row).lower():
                df = df_temp
            else:
                df = {}
                key_array = []
                for key in df_temp.keys():
                    key_array.append(key)
                df['Frequency [Hz]'] = df_temp[key_array[0]]
                df['Phase'] = df_temp[key_array[3]]
                if 'vswr' in mode:
                    df['VSWR'] = df_temp[key_array[1]]
                elif 'rxpath' in mode:
                    df['Magnitude'] = df_temp[key_array[1]]

        elif vna == 'mdo3034':
            df = pd.read_csv(file)
            skiprows = 0
            for index, row in df.iterrows():
                skiprows += 1
                if 'fd max' in str(row).lower():
                    skiprows += 1
                    break

            df = pd.read_csv(file, skiprows=skiprows, encoding='cp1252')

        else:
            print(f'Unknown VNA {vna}: Exiting...')
            exit(1)
        
        keys = list(df.keys())
        freq = None
        vswr = None
        magnitude = None
        phase = None
        for key in keys:
            if not isinstance(df[key], (int,float)):
                pass
            # if 'unnamed' in key.lower():  # Break from loop after the first ZVH sweep.
            #     verbose and print('\t-end of first sweep')
            #     break
            if 'freq' in key.lower():
                freq = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-FREQUENCY data found in: {name}')
            if 'vswr' in key.lower():
                vswr = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-VSWR data found in: {name}')
            if 'mag' in key.lower() or 'fd max' in key.lower():
                magnitude = pd.to_numeric(df[key], errors='coerce')
                verbose and print(f'\t-MAGNITUDE data found in: {name}')
            if 'pha' in key.lower():
                # phase = pd.to_numeric(df[key], errors='coerce')
                phase = df[key]
                phase = (phase + 180) % 360 - 180  # Ensure phase values are between -180 to 180 degrees
                phase_unwrapped = np.unwrap(phase, period=360)
                verbose and print(f'\t-PHASE data found in: {name}')

        data = RSData(name=name, freq=freq, vswr=vswr, magnitude=magnitude, phase=phase, phase_unwrapped=phase_unwrapped)
        all_data.names.append(name)
        all_data.datas.append(data)

    return all_data


def plot_rxpath(data, directory=''):
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

    # Make lines in a combination of the following line styles and colours, so there are 40 different line combos
    # Set line style as follows: linestyle=LINE_STYLES[int(index/NUM_COLOURS)]
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_COLOURS = 10

    outfile = f'{directory}rxpath_{data.site_code}_{data.date}.png'
    mean_magnitude = 0.0
    mean_phase = 0.0
    total_antennas = 0.0
    xmin = data.datas[0].freq[0]
    xmax = data.datas[0].freq[0]
    ymin = data.datas[0].magnitude[0]
    ymax = data.datas[0].magnitude[0]
    max_mag = data.datas[0].magnitude
    min_mag = data.datas[0].magnitude
    max_pha = data.datas[0].phase_unwrapped
    min_pha = data.datas[0].phase_unwrapped

    fig, ax = plt.subplots(2, 1, figsize=[13, 8])
    fig.suptitle(f'{data.vna.upper()} Data: RX Path per Antenna\n{data.site_name} {data.date}')
    for index, name in enumerate(data.names):
        mean_magnitude += data.datas[index].magnitude
        mean_phase += data.datas[index].phase
        total_antennas += 1.0
        # Determine the y and x limits
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        if np.min(data.datas[index].magnitude) < ymin:
            ymin = np.min(data.datas[index].magnitude)
        if np.max(data.datas[index].magnitude) > ymax:
            ymax = np.max(data.datas[index].magnitude)

        max_mag = np.maximum(max_mag, data.datas[index].magnitude)
        min_mag = np.minimum(min_mag, data.datas[index].magnitude)
        max_pha = np.maximum(max_pha, data.datas[index].phase_unwrapped)
        min_pha = np.minimum(min_pha, data.datas[index].phase_unwrapped)
        

        # Scale freq by 1E+6 to make x-axis units MHz instead of Hz
        ax[0].plot(data.datas[index].freq/1E+6, data.datas[index].magnitude, label=data.datas[index].name,
                    linestyle=LINE_STYLES[int(index/NUM_COLOURS)])
        ax[1].plot(data.datas[index].freq/1E+6, data.datas[index].phase, label=data.datas[index].name,
                    linestyle=LINE_STYLES[int(index/NUM_COLOURS)])

    # Plot variation on right axis
    ax0_r = ax[0].twinx()
    ax0_r.plot(data.datas[0].freq/1E+6, max_mag - min_mag, color='tab:blue')
    ax0_r.set_ylabel('Max Magnitude Variation (dB)', color='tab:blue')
    # ax0_r.set_ylim([0, 10])

    ax1_r = ax[1].twinx()
    ax1_r.plot(data.datas[0].freq/1E+6, max_pha - min_pha, color='tab:blue')
    ax1_r.set_ylabel('Max Phase Variation [°]', color='tab:blue')
    # ax1_r.set_ylim([0, 50])

    mean_magnitude /= total_antennas
    mean_phase /= total_antennas
    ax[0].plot(data.datas[0].freq/1E+6, mean_magnitude, '--k', label='mean')
    ax[1].plot(data.datas[0].freq/1E+6, mean_phase, '--k', label='mean')

    # Change x limits to MHz from Hz
    xmin /= 1E+6
    xmax /= 1E+6
    # Set base plot limits for magnitude plot
    if ymin > 20:
        ymin = 20 # dB
    if ymax < 30:
        ymax = 30 # dB

    ax[0].legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim([xmin, xmax])
    ax[1].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
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

    outfile = f'{directory}vswr_{data.site_code}_{data.date}.png'
    mean_vswr = 0.0
    total_antennas = 0.0
    xmin = 8.0e6  # Hz
    xmax = 20.0e6 # Hz

    plt.figure(figsize=[13, 8])
    plt.suptitle(f'{data.vna.upper()} Data: VSWR per Antenna\n{data.site_name} {data.date}')
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

    xmin /= 1E+6  # MHz
    xmax /= 1E+6  # MHz

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


def plot_skynoise(data, directory=''):
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

    outfile = f'{directory}sky_noise_{data.site_code}_{data.date}.png'
    mean_magnitude = 0.0
    total_antennas = 0.0
    xmin = 20.0e6
    xmax = 0.0

    fig, ax = plt.subplots(1, 1, figsize=[13, 8])
    fig.suptitle(f'{data.vna.upper()} Data: Sky Noise\n{data.site_name} {data.date}')
    for index, name in enumerate(data.names):
        mean_magnitude += data.datas[index].magnitude
        total_antennas += 1.0
        if np.min(data.datas[index].freq) < xmin:
            xmin = np.min(data.datas[index].freq)
        if np.max(data.datas[index].freq) > xmax:
            xmax = np.max(data.datas[index].freq)
        ax.plot(data.datas[index].freq, data.datas[index].magnitude, label=data.datas[index].name)

    mean_magnitude /= total_antennas
    ax.plot(data.datas[0].freq, mean_magnitude, '--k', label='mean magnitude')

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
    parser = argparse.ArgumentParser(description='SuperDARN Canada© -- Engineering Diagnostic Tools Kit: Data Plotting'
                                                 'Given a set of CSV files this program will generate comparison plots'
                                                 'for engineering diagnostics.')
    parser.add_argument('-s', '--site', type=str, help='Radar code of the site this data is from, eg: sas, pgr, rkn...')
    parser.add_argument('-d', '--directory', type=str, default='', help='Directory containing CSV files with data to be plotted.')
    parser.add_argument('-o', '--outdir', type=str, default='', help='Directory to save output plots.')
    parser.add_argument('-p', '--pattern', type=str, default='', help='File naming pattern (eg. sas-vswr-, pgr-rxpath-).')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['vswr','rxpath','skynoise'],
                        help='Select the type of plot to make: (vswr, rxpath, skynoise).')
    parser.add_argument('-t', '--device', type=str, required=True, choices=['zvh4','trvna','mdo3034'],
                        help='Measurement device that collected the data to plot. Options (zvh4, trvna, mdo3034)')
    parser.add_argument('--date', type=str, default='1970-01-01', help='date of the data to be plotted (yyyy-mm-dd)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information.')

    args = parser.parse_args()

    radar_codes = {
        "sas" : "Saskatoon",
        "pgr" : "Prince George",
        "rkn" : "Rankin Inlet",
        "inv" : "Inuvik",
        "cly" : "Clyde River",
        "lab" : "SuperDARN Lab"
    }

    site_code = args.site.lower()
    if site_code in radar_codes:
        site_name = radar_codes[site_code]
    else:
        site_name = args.site
    
    directory = args.directory
    outdir = args.outdir
    if outdir == '':
        outdir = directory
    pattern = args.pattern
    date = args.date
    verbose = args.verbose
    device = args.device
    mode = args.mode


    data = read_data(directory, pattern, date, verbose, site_code, site_name, device, mode)

    if args.mode == 'vswr':
        plot_vswr(data, directory=outdir)
    elif args.mode == 'rxpath':
        plot_rxpath(data, directory=outdir)
    elif args.mode == "skynoise":
        plot_skynoise(data, directory=outdir)
    else:
        print('Select a mode: vswr, rxpath, or skynoise') 

    return None


if __name__ == '__main__':
    main()
