import collections
import glob
import argparse
import os

from analyze_times import plot_single_param, full_data_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", help="Name of the file to analyze.", required=True)
    parser.add_argument("-s", "--timestamp", help="Timestamp to identify which log files to scrub. "
                                                  "Format as YYYY.MM.DD.HH:MM, or truncate to a desired "
                                                  "precision. All files with this prefix will be analyzed.",
                        required=True)
    parser.add_argument("-V", "--version", help="Version of Borealis which generated the log file. Supported"
                                                "versions are 'v0.5' and 'v0.6'", default='v0.5')
    parser.add_argument("-d", "--plot_dir", help="Directory to save plots.")

    args = parser.parse_args()

    if args.version not in ['v0.5', 'v0.6']:
        raise ValueError("Unknown Borealis version")

    if args.plot_dir is not None:
        plot_dir = args.plot_dir
    else:
        plot_dir = args.input_directory

    filetypes_to_datatypes = {
        'radar_control': ['num_sqns'],
        'data_write': ['parse_time', 'write_time'],
        'signal_processing': ['memcpy_time', 'dec_time', 'bf_time', 'corr_time', 'send_time'],
        'rx_signal_processing': ['memcpy_time', 'processing_time', 'send_time']
    }

    # Dictionary of {timestamp: filelist} pairs for all matching files. Essentially just grouping concurrent logs.
    grouped_files = collections.defaultdict(list)

    for file in glob.glob("{}/{}*".format(args.input_directory, args.timestamp)):
        filetype = file.split('-')[-1]

        if filetype not in filetypes_to_datatypes.keys():
            continue

        # for data_type in filetypes_to_datatypes[filetype]:
        #     plot_single_param(file, data_type, args.version, plot_dir)

        name = os.path.basename(file)
        tstamp = name[:16]
        grouped_files[tstamp].append(file)

    for tstamp, filelist in grouped_files.items():
        print("Plotting {}".format(tstamp))
        full_data_plot(tstamp, filelist, filetypes_to_datatypes)

