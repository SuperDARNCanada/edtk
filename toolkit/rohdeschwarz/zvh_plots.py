"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: Data Plotting

Author: Adam Lozinsky, Theodore Kolkman, Saif Marei 
Date: December 10, 2024 
Affiliation: University of Saskatchewan

SuperDARN engineers make a series of measurements for each antennas RF path using a Rohde & Schwarz
ZVH or Copper Mountain TR VNA. These measurements can be converted into or recorded as .csv files.
The files contain different data based on the instrument settings, but it is per antenna. It is
preferred to plot all the data for each antenna on one plot so differences and outliers are easily
visible. This tool will produce those common plots from the .csv files.

Use 'python zvh_plots.py --help' to discover options if running directly from command line.
"""

from dataclasses import dataclass, field
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Optional


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
    device: str = field(default_factory=str)
    names: list = field(default_factory=list)
    datas: list = field(default_factory=list)


def read_data(
        directory: str, 
        site_code: str, 
        site_name: str, 
        device: str, 
        mode: str, 
        pattern: Optional[str] = '', 
        date: Optional[str] = '', 
        verbose: Optional[bool] = False, 
        filter_list: Optional[List[str]] = ''
    ) -> RSAllData:
    """
    Load the VNA data from .csv files from either a parent directory given a file pattern or from a
    directory directly. The data is then loaded into a dataclass and returned for further
    processing.
    
    For Rohde & Schwarz ZVH, data must be read in from the specific R&S format. For Copper Mountain
    VNA, data can be read in directly from utf-8 CSV files

    Parameters
    ----------
        directory : str
            The directory or parent directory containing the .csv files.
        pattern : str
            The file naming pattern of files to load; eg. rkn_vswr would yield all rkn_vswr*.csv in
            directory tree.
        date : str
            Recording date of data to be plotted in the form yyyy-mm-dd
        verbose : bool
            True will print more information about whats going on, False squelches.
        site : str
            Name of the site the data was taken from; used in naming plots and plot titles.
        device : str
            Vector network analyzer that produced the data to be plotted. Options are 'zvh' for the
            Rohde & Schwarz ZVH, or 'trvna' for the Copper Mountain TR VNA
        mode : str
            The type of data collected. Options are vswr, rxpath, skynoise

    Returns
    -------
        all_data : dataclass
            A dataclass containing all the data for each antenna from the Rohde & Schwarz .csv
            files.
    """

    files = glob.glob(f"{directory}/*/{pattern}*.csv")
    if files == []:
        files = sorted(glob.glob(f"{directory}/{pattern}*.csv"))
    if verbose:
        print(f"Following files found in {directory}:\n", [os.path.basename(f) for f in files])

    filtered_files = []
    print(filter_list)
    for filter in filter_list:
        for file in files:
            if filter in os.path.basename(file):
                filtered_files.append(file)
                if verbose:
                    print(f"Filter removed file: {file}")
    files = [f for f in files if f not in filtered_files]

    all_data = RSAllData()
    all_data.site_code = site_code
    all_data.site_name = site_name
    all_data.device = device
    all_data.date = date

    for file in files:
        name = os.path.basename(file).replace('.csv', '')
        if verbose: 
            print(f'loading file: {file}')
        if device == 'zvh4':
            # Determine which row the data starts on
            df = pd.read_csv(file, encoding='cp1252')
            skiprows = 0
            endrow = 0
            for index, row in df.iterrows():
                skiprows += 1
                # ZVH4 records specific time data was collected. Use this for the date instead.
                if 'date' in str(row).lower():
                    try:
                        date = datetime.strptime(row.iloc[1], "%m/%d/%Y")
                    except:
                        date = datetime.strptime(row.iloc[1], "%Y-%m-%d")
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

            # The ZVH .csv files are in format cp1252 not utf-8 so using utf-8 will break on degrees
            # symbol.
            df = pd.read_csv(file, skiprows=skiprows, nrows=endrow-skiprows, encoding='cp1252')

        elif device == 'trvna':
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

        elif device == 'mdo3034':
            df = pd.read_csv(file)
            skiprows = 0
            for index, row in df.iterrows():
                skiprows += 1
                if 'fd max' in str(row).lower():
                    skiprows += 1
                    break

            df = pd.read_csv(file, skiprows=skiprows, encoding='cp1252')

        else:
            print(f'Unknown test device "{device}": Exiting...')
            exit(1)
        
        keys = list(df.keys())
        freq = None
        vswr = None
        magnitude = None
        phase = None
        phase_unwrapped = None
        for key in keys:
            if not isinstance(df[key], (int,float)):
                pass
            # ZVH4 files contain duplicates of data - only record it the first time.
            if 'freq' in key.lower() and freq is None:
                freq = pd.to_numeric(df[key], errors='coerce')
                if verbose:
                    print(f'\t-FREQUENCY data found in: {name}')
            if 'vswr' in key.lower() and vswr is None:
                vswr = pd.to_numeric(df[key], errors='coerce')
                if verbose:
                    print(f'\t-VSWR data found in: {name}')
            if ('mag' in key.lower() or 'fd max' in key.lower()) and magnitude is None:
                magnitude = pd.to_numeric(df[key], errors='coerce')
                if verbose:
                    print(f'\t-MAGNITUDE data found in: {name}')
            if 'pha' in key.lower() and phase is None:
                phase = pd.to_numeric(df[key], errors='coerce')
                phase = (phase + 180) % 360 - 180  # Ensure phase values are between -180 to 180 deg
                phase_unwrapped = pd.Series(np.unwrap(phase, period=360))
                if verbose:
                    print(f'\t-PHASE data found in: {name}')

        data = RSData(name=name, freq=freq, vswr=vswr, magnitude=magnitude, phase=phase, 
                      phase_unwrapped=phase_unwrapped)
        all_data.names.append(name)
        all_data.datas.append(data)

    return all_data


def calculate_ticks(ax, ticks, round_to=0.1, center=False):
    """
    From https://stackoverflow.com/questions/20243683/align-twinx-tick-marks
    """
    upperbound = np.ceil(ax.get_ybound()[1]/round_to)
    lowerbound = np.floor(ax.get_ybound()[0]/round_to)
    dy = upperbound - lowerbound
    fit = np.floor(dy/(ticks - 1)) + 1
    dy_new = (ticks - 1)*fit
    if center:
        offset = np.floor((dy_new - dy)/2)
        lowerbound = lowerbound - offset
    values = np.linspace(lowerbound, lowerbound + dy_new, ticks)
    return values*round_to


def plot_rxpath(data: RSAllData, directory: str = '', filename: str = '', plot_stats: bool = False):
    """
    Create a plot of magnitude and phase vs frequency for each antenna receive path. Optionally, a
    third plot showing variance stats can be plotted, showing the Root Mean Square Error (RMSE) and
    Mean Absolute Deviation (MAD) for both magnitude and phase.
    
    Parameters
    ----------
        data : RSAllData dataclass
            A dataclass containing measured data from a ZVH4 or TRVNA. Data must contain frequency,
            magnitude, and phase data.
        directory : str
            The output file directory to save the plot in.
        filename : str
            The output file name to write the plot to. Defaults to rxpath_[Site ID]_YYYY-MM-DD.png
        plot_stats : bool
            If true, adds a third plot showing variance stats for the magnitude and phase data. If
            false, only magnitude and phase will be plotted.
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

    # Make lines in a combination of the following line styles and colours, so there are 40
    # different line combos. Set line style as follows:
    # linestyle=LINE_STYLES[int(index/NUM_COLOURS)]
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_COLOURS = 10

    if filename == '':
        outfile = f'{directory}rxpath_{data.site_code}_{data.date}.png'
    else:
        outfile = f'{directory}{filename}'
    if plot_stats:
        fig, ax = plt.subplots(3, 1, figsize=[13, 10])
    else: 
        fig, ax = plt.subplots(2, 1, figsize=[13, 8])
    fig.suptitle(f'{data.device.upper()} Data: Receive Path Amplification per Antenna\n'
                 f'{data.site_name} {data.date}')

    # Construct 2D arrays for each data type for stat calculations.
    frequency_alldata = np.array([d.freq for d in data.datas])
    magnitude_alldata = np.array([d.magnitude for d in data.datas])
    phase_unwrapped_alldata = np.array([d.phase_unwrapped for d in data.datas])

    for index, name in enumerate(data.names):
        # Scale freq by 1E+6 to make x-axis units MHz instead of Hz
        ax[0].plot(data.datas[index].freq/1E+6, data.datas[index].magnitude, label=name,
                    linestyle=LINE_STYLES[int(index/NUM_COLOURS)])
        ax[1].plot(data.datas[index].freq/1E+6, data.datas[index].phase, label=name,
                    linestyle=LINE_STYLES[int(index/NUM_COLOURS)])

    # Plot average mag and phase
    mean_magnitude = np.mean(magnitude_alldata, 0)
    mean_phase_unwrapped = np.mean(phase_unwrapped_alldata, 0) # Calculate mean with unwrapped data
    mean_phase = (mean_phase_unwrapped + 180) % 360 - 180      # Plot the phase mean wrapped
    ax[0].plot(data.datas[0].freq/1E+6, mean_magnitude, '--k', label='mean')
    ax[1].plot(data.datas[0].freq/1E+6, mean_phase, '--k', label='mean')


    # Define plot limits. Phase is wrapped, so limits are fixed.
    xmin = np.min(frequency_alldata) / 1E+6 # / 1E+6 to change from Hz to MHz
    xmax = np.max(frequency_alldata) / 1E+6
    ymin = np.round(np.min(magnitude_alldata) - 1) # Adjust limits to add whitespace above/below data
    ymax = np.round(np.max(magnitude_alldata) + 1)


    # Set base plot limits for magnitude plot
    if ymin > 20:
        ymin = 20 # dB
    if ymax < 30:
        ymax = 30 # dB


    if plot_stats:
        # Plot data variance statistics
        rmse_mag = np.sqrt(np.mean(np.square(magnitude_alldata - mean_magnitude), 0))
        rmse_pha = np.sqrt(np.mean(np.square(phase_unwrapped_alldata - mean_phase_unwrapped), 0))
        mad_mag = np.mean(np.abs(magnitude_alldata - mean_magnitude), 0)
        mad_pha = np.mean(np.abs(phase_unwrapped_alldata - mean_phase_unwrapped), 0)

        ax[2].set_title('Magnitude and Phase Variation')
        ax2_left = ax[2]
        ax2_left.plot(data.datas[0].freq/1E+6, rmse_mag, color='tab:blue', linestyle='--', 
                      label='Root Mean Squared Error (RMSE)')
        ax2_left.plot(data.datas[0].freq/1E+6, mad_mag, color='tab:blue', linestyle='-', 
                      label='Mean Absolute Deviation (MAD)')
        ax2_left.set_ylabel('Magnitude [dB]', color='tab:blue')

        ax2_right = ax[2].twinx()
        ax2_right.plot(data.datas[0].freq/1E+6, rmse_pha, color='tab:red', linestyle='--', 
                       label='Root Mean Squared Error (RMSE)')
        ax2_right.plot(data.datas[0].freq/1E+6, mad_pha, color='tab:red', linestyle='-', 
                       label='Mean Absolute Deviation (MAD)')
        ax2_right.set_ylabel('Phase [°]', color='tab:red')

        ax[2].set_xlabel('Frequency [MHz]')

        ax2_left.set_ylim(bottom=0)
        ax2_right.set_ylim(bottom=0)
        ax2_left.set_yticks(calculate_ticks(ax2_left, 5, 0.5))
        ax2_right.set_yticks(calculate_ticks(ax2_right, 5, 5))
        ax2_left.tick_params(axis='y', colors='tab:blue')
        ax2_right.tick_params(axis='y', colors='tab:red')
        ax2_left.set_xlim([xmin, xmax])
        ax2_right.set_xlim([xmin, xmax])
        ax2_left.legend(loc='upper left')
        ax2_right.legend(loc='upper right')
        ax2_left.grid()


    ax[0].legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim([xmin, xmax])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_yticks([180,90,0,-90,-180])
    ax[0].set_ylim([ymin, ymax])
    ax[1].set_xlabel('Frequency [MHz]')
    ax[0].set_ylabel('Magnitude [dB]')
    ax[1].set_ylabel('Phase [°]')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'rx path file created at: {outfile}')
    return


def plot_vswr(data: RSAllData, directory: str = '', filename: str = '', plot_stats: bool = False):
    """
    Create a plot of voltage standing wave ration (VSWR) and phase vs frequency for each antenna.
    Optionally, a third plot showing variance stats can be plotted, showing the Root Mean Square
    Error (RMSE) and Mean Absolute Deviation (MAD) for both VSWR and phase.
    
    Parameters
    ----------
        data : RSAllData dataclass
            A dataclass containing measured data from a ZVH4 or TRVNA. Data must contain frequency,
            VSWR, and phase data.
        directory : str
            The output file directory to save the plot in.
        filename : str
            The output file name to write the plot to. Defaults to vswr_[Site ID]_YYYY-MM-DD.png
        plot_stats : bool
            If true, adds a third plot showing variance stats for the VSWR and phase data. If false,
            only VSWR and phase will be plotted.
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

    if filename == '':
        outfile = f'{directory}vswr_{data.site_code}_{data.date}.png'
    else:
        outfile = f'{directory}{filename}'
    if plot_stats:
        fig, ax = plt.subplots(3, 1, figsize=[13, 10])
    else: 
        fig, ax = plt.subplots(2, 1, figsize=[13, 8])
    fig.suptitle(f'{data.device.upper()} Data: Voltage Standing Wave Ratio (VSWR) per Antenna\n'
                 f'{data.site_name} {data.date}')
    
    # Construct 2D arrays for each data type for stat calculations.
    frequency_alldata = np.array([d.freq for d in data.datas])
    vswr_alldata = np.array([d.vswr for d in data.datas])
    phase_unwrapped_alldata = np.array([d.phase_unwrapped for d in data.datas])

    for index, name in enumerate(data.names):
        ax[0].plot(data.datas[index].freq/1E+6, data.datas[index].vswr, label=name,
                   linestyle=LINE_STYLES[int(index/NUM_COLOURS)])
        ax[1].plot(data.datas[index].freq/1E+6, data.datas[index].phase, label=name,
                   linestyle=LINE_STYLES[int(index/NUM_COLOURS)])

    mean_vswr = np.mean(vswr_alldata, 0)
    mean_phase_unwrapped = np.mean(phase_unwrapped_alldata, 0) # Calculate mean with unwrapped data
    mean_phase = (mean_phase_unwrapped + 180) % 360 - 180      # Plot the phase mean wrapped
    ax[0].plot(data.datas[0].freq/1E+6, mean_vswr, '--k', label='mean')
    ax[1].plot(data.datas[0].freq/1E+6, mean_phase, '--k', label='mean')


    # Define plot limits. Phase is wrapped, so limits are fixed.
    xmin = np.min(frequency_alldata) / 1E+6 # / 1E+6 to change from Hz to MHz
    xmax = np.max(frequency_alldata) / 1E+6
    ymin = 1                                    # Minimum VSWR is 1
    ymax = np.ceil(np.max(vswr_alldata))   # Adjust limits to add whitespace above/below data

    # Set base plot limits for vswr plot
    if ymax < 3:
        ymax = 3 # dB

    if plot_stats:
        # Plot data variance statistics
        rmse_vswr = np.sqrt(np.mean(np.square(vswr_alldata - mean_vswr), 0))
        rmse_pha = np.sqrt(np.mean(np.square(phase_unwrapped_alldata - mean_phase_unwrapped), 0))
        mad_vswr = np.mean(np.abs(vswr_alldata - mean_vswr), 0)
        mad_pha = np.mean(np.abs(phase_unwrapped_alldata - mean_phase_unwrapped), 0)

        ax[2].set_title('VSWR and Phase Variation')
        ax2_left = ax[2]
        ax2_left.plot(data.datas[0].freq/1E+6, rmse_vswr, color='tab:blue', linestyle='--', 
                      label='Root Mean Squared Error (RMSE)')
        ax2_left.plot(data.datas[0].freq/1E+6, mad_vswr, color='tab:blue', linestyle='-', 
                      label='Mean Absolute Deviation (MAD)')
        ax2_left.set_ylabel('VSWR', color='tab:blue')

        ax2_right = ax[2].twinx()
        ax2_right.plot(data.datas[0].freq/1E+6, rmse_pha, color='tab:red', linestyle='--', 
                       label='Root Mean Squared Error (RMSE)')
        ax2_right.plot(data.datas[0].freq/1E+6, mad_pha, color='tab:red', linestyle='-', 
                       label='Mean Absolute Deviation (MAD)')
        ax2_right.set_ylabel('Phase [°]', color='tab:red')

        ax[2].set_xlabel('Frequency [MHz]')

        ax2_left.set_ylim(bottom=0)
        ax2_right.set_ylim(bottom=0)
        ax2_left.set_yticks(calculate_ticks(ax2_left, 5, 0.1))
        ax2_right.set_yticks(calculate_ticks(ax2_right, 5, 5))
        ax2_left.tick_params(axis='y', colors='tab:blue')
        ax2_right.tick_params(axis='y', colors='tab:red')
        ax2_left.set_xlim([xmin, xmax])
        ax2_right.set_xlim([xmin, xmax])
        ax2_left.legend(loc='upper left')
        ax2_right.legend(loc='upper right')
        ax2_left.grid()


    ax[0].legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim([xmin, xmax])
    ax[1].set_xlim([xmin, xmax])
    ax[0].locator_params(axis='y', nbins=4)
    ax[1].set_yticks([180,90,0,-90,-180])
    ax[0].set_ylim([ymin, ymax])
    ax[1].set_xlabel('Frequency [MHz]')
    ax[0].set_ylabel('VSWR')
    ax[1].set_ylabel('Phase [°]')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'VSWR plot created at: {outfile}')
    return


def plot_skynoise(data: RSAllData, directory: str = '', filename: str = '', plot_stats: bool = False):
    """
    Create a spectrum plot showing sky noise as power vs frequency. Optionally, a second plot can be
    added to show variance stats in the skynoise datasets plotted.
    
    Parameters
    ----------
        data : RSAllData dataclass
            A dataclass containing measured data from a ZVH4 or MDO3034. Data must contain frequency
            and magnitude data.
        directory : str
            The output file directory to save the plot in.
        filename : str
            The output file name to write the plot to. Defaults to skynoise_[Site ID]_YYYY-MM-DD.png
        plot_stats : bool
            If true, adds a second plot showing variance stats for the magnitude data. If false,
            only magnitude will be plotted.
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

    # Make lines in a combination of the following line styles and colours, so there are 40
    # different line combos. Set line style as follows:
    # linestyle=LINE_STYLES[int(index/NUM_COLOURS)]
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_COLOURS = 10

    if filename == '':
        outfile = f'{directory}skynoise_{data.site_code}_{data.date}.png'
    else:
        outfile = f'{directory}{filename}'
    if plot_stats:
        fig, ax = plt.subplots(2, 1, figsize=[13, 10])
    else: 
        fig, ax = plt.subplots(1, 1, figsize=[13, 8])
    fig.suptitle(f'{data.device.upper()} Data: Sky Noise\n'
                 f'{data.site_name} {data.date}')
    
    # Construct 2D arrays for each data type for stat calculations.
    frequency_alldata = np.array([d.freq for d in data.datas])
    magnitude_alldata = np.array([d.magnitude for d in data.datas])

    for index, name in enumerate(data.names):
        # Scale freq by 1E+6 to make x-axis units MHz instead of Hz
        ax[0].plot(data.datas[index].freq/1E+6, data.datas[index].magnitude, label=name,
                   linestyle=LINE_STYLES[int(index/NUM_COLOURS)])

    mean_magnitude = np.mean(magnitude_alldata, 0)
    ax[0].plot(data.datas[0].freq/1E+6, mean_magnitude, '--k', label='mean')

    # Define plot limits. Phase is wrapped, so limits are fixed.
    xmin = np.min(frequency_alldata) / 1E+6 # / 1E+6 to change from Hz to MHz
    xmax = np.max(frequency_alldata) / 1E+6
    ymin = np.round(np.min(magnitude_alldata) - 1) # Adjust limits to add whitespace above/below data
    ymax = np.round(np.max(magnitude_alldata) + 1)

    # Set base plot limits for magnitude plot
    base_ymin = -100
    base_ymax = -50
    if ymin > base_ymin:
        ymin = base_ymin # dB
    if ymax < base_ymax:
        ymax = base_ymax # dB

    if plot_stats:
        # Plot data variance statistics
        rmse_mag = np.sqrt(np.mean(np.square(magnitude_alldata - mean_magnitude), 0))
        mad_mag = np.mean(np.abs(magnitude_alldata - mean_magnitude), 0)

        ax[1].set_title('Magnitude Variation')
        ax[1].plot(data.datas[0].freq/1E+6, rmse_mag, color='tab:blue', linestyle='--', 
                   label='Root Mean Squared Error (RMSE)')
        ax[1].plot(data.datas[0].freq/1E+6, mad_mag, color='tab:blue', linestyle='-', 
                   label='Mean Absolute Deviation (MAD)')
        ax[1].set_ylabel('Power [dBm]', color='tab:blue')

        ax[1].set_xlabel('Frequency [MHz]')

        ax[1].set_ylim(bottom=0)
        ax[1].tick_params(axis='y', colors='tab:blue')
        ax[1].set_xlim([xmin, xmax])
        ax[1].legend(loc='upper left')
        ax[1].grid()

    ax[0].legend(loc='center', fancybox=True, ncol=7, bbox_to_anchor=[0.5, -0.4])
    ax[0].grid()
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_xlabel('Frequency [MHz]')
    ax[0].set_ylabel('Power [dBm]')
    plt.tight_layout()
    plt.savefig(outfile)

    print(f'Skynoise file created at: {outfile}')
    return


def main():
    parser = argparse.ArgumentParser(
            description='SuperDARN Canada© -- Engineering Diagnostic Tools Kit: Data Plotting\n\n'
                        'Given a set of CSV files, this program will generate comparison plots '
                        'for engineering diagnostics.',
            formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, 
                                                                              max_help_position=50, 
                                                                              width=100)
    )
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Print more information.')
    parser.add_argument('--site', type=str, required=True,
                        help='Radar code of the site this data is from, eg: sas, pgr, rkn...')
    parser.add_argument('--directory', type=str, required=True,
                        help='Directory containing CSV files with data to be plotted.')
    parser.add_argument('--mode', type=str, required=True, choices=['vswr','rxpath','skynoise'],
                        help='Select the type of plot to make: (vswr, rxpath, skynoise).')
    parser.add_argument('--device', type=str, required=True, choices=['zvh4','trvna','mdo3034'],
                        help='Measurement device that collected the data to plot. Options (zvh4, '
                             'trvna, mdo3034)')
    parser.add_argument('--outdir', type=str, default='', 
                        help='Directory to save output plots.')
    parser.add_argument('--filename', type=str, default='',
                        help='Specify the filename of the output plot.')
    parser.add_argument('--pattern', type=str, default='', 
                        help='File naming pattern (eg. sas-vswr-, pgr-rxpath-).')
    parser.add_argument('--filter', type=str, nargs='+', 
                        help='Filter out files found in the specified directory by listing any '
                             'number of expressions to be filtered out. Example: "--filter 18 19" '
                             'will omit files containing 18 and 19 in the filename.')
    parser.add_argument('--date', type=str, default='1970-01-01', 
                        help='date of the data to be plotted (yyyy-mm-dd)')
    parser.add_argument('--plot_stats', action='store_true', 
                        help='Adds an extra plot showing the variance in the magnitude/vswr and '
                             'phase across all plotted data.')

    args = parser.parse_args()

    radar_codes = {
        "sas" : "Saskatoon",
        "pgr" : "Prince George",
        "rkn" : "Rankin Inlet",
        "inv" : "Inuvik",
        "cly" : "Clyde River",
        "lab" : "SuperDARN Lab"
    }
    vna_devices = ['zvh4', 'trvna']
    spectrum_devices = ['zvh4', 'mdo3034']

    site_code = args.site.lower()
    site_name = radar_codes.get(site_code, args.site)
    
    directory = args.directory
    outdir = args.outdir
    if outdir == '':
        outdir = directory
    filename = args.filename
    pattern = args.pattern
    date = args.date
    verbose = args.verbose
    device = args.device
    mode = args.mode
    if (mode in ['vswr', 'rxpath'] and device not in vna_devices) or \
                        (mode in ['skynoise'] and device not in spectrum_devices):
        print(f'Error: device specified ({device}) cannot collect {mode} data')
    if args.filter is None:
        filter_list = []
    else:
        filter_list = args.filter
    plot_stats = args.plot_stats


    data = read_data(directory=directory, pattern=pattern, date=date, verbose=verbose, 
                     site_code=site_code, site_name=site_name, device=device, mode=mode, 
                     filter_list=filter_list)

    if args.mode == 'vswr':
        plot_vswr(data, directory=outdir, filename=filename, plot_stats=plot_stats)
    elif args.mode == 'rxpath':
        plot_rxpath(data, directory=outdir, filename=filename, plot_stats=plot_stats)
    elif args.mode == "skynoise":
        plot_skynoise(data, directory=outdir, filename=filename, plot_stats=plot_stats)
    else:
        print('Select a mode: vswr, rxpath, or skynoise') 

    return None


if __name__ == '__main__':
    main()
