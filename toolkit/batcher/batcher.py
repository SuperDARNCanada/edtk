"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (Batcher)

Author: Adam Lozinsky
Date: December 8, 2021
Affiliation: University of Saskatchewan

The SuperDARN data is stored in HDF5 and DMAP formates in calendar heirachical format. Often there is a need to
retreive a list of the data files with their absolute paths as well as each files structure (site, array, iqdat, dmap)
and their type (antennas_iq, bfiq, rawacf). This small package fetches that information and returns a list of tuples.
This is usefull for batch loading various processes such as 'convert', 'fixer', 'pydarnio', or 'plotting'.

Use 'python batcher.py --help' to discover options if running directly from command line.
"""
from dataclasses import dataclass
import argparse
import glob
import os
import h5py


@dataclass(frozen=True, order=True)
class FileData:
    """Data class for keeping the Rohde & Schwarz ZVH data."""
    path: str
    structure: float
    type: float
    averaging: str
    year: int
    month: int
    day: int
    hour: int


def get_batch(directory, pattern='*', verbose=False):
    """
    Parameters
    ----------
        directory : str
            The directory or parent directory containing the .csv files.
        pattern : str
            The file naming pattern of files to load; eg. rkn_vswr would yield all rkn_vswr*.csv in directory tree.
        verbose : bool
            True will print more information about whats going on, False squelches.

    Returns
    -------
    """

    files = glob.glob(directory + '/**/' + pattern, recursive=True)
    if files == []:
        files = glob.glob(directory + pattern)
        if files == []:
            print(f'no files found at: {directory} or')
            print(f'no files found with pattern: {pattern}')
            exit()
    verbose and print("files found:\n", files)

    return files


def parse_files(files):
    for file in files:
        f = h5py.File(file, 'r')
        keys = list(f.keys())
        print(keys)
        print(f.attrs.keys())
        for key in keys:
            print(f[key].attrs.keys())


    return


def _print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return None


def walk_hdf5(filepath):
    """
    Walks the tree for a given hdf5 file.

    Parameters
    ----------
        filepath : string
            File path and name to hdf5 file to be walked.

    Returns
    -------
        None
    """

    f = h5py.File(filepath, 'r')
    f.visititems(_print_attrs)
    return None

def query():
    # fetch.py -d '/data' --query="type=rawacf,sturcture=array,cpid=3503,..." |

if __name__ == '__main__':
    """
    This only needs to create the metadata and stuff for the things fetched.
    It should probably be mature enough to include the conversion tuples information.
    Looks like Carley has some sort of metadata database created so this is like a mini database
    """
    # walk_hdf5('/home/glatteis/SuperDARN/data/20210225.2219.22.sas.0.rawacf.hdf5.array')
    parse_files(['/home/glatteis/SuperDARN/data/20210225.2219.22.sas.0.rawacf.hdf5.array', '/home/glatteis/SuperDARN/data/20210225.2219.22.sas.0.rawacf.hdf5.site'])
