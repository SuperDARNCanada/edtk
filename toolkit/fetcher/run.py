"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (Fetcher)

Author: Adam Lozinsky
Date: December 15, 2021
Affiliation: University of Saskatchewan

Given a search query (--query) or file name patter (--pattern) and parent directory returns a list of data files.
"""

import os
import glob
import argparse
from dataclasses import dataclass
from query import Query
from fetch import Fetch
from batch import Batch


@dataclass(frozen=True, order=True)
class FileData:
    """Data class for keeping the data file queryable data."""
    filename: str = "file.this.2020202.is.fake"
    path: str = "/path/to/the/directory/and/the/file.this.2020202.is.fake"
    version: str = "4.0"
    structure: str = "array"
    type: str = "rawacf"
    year: int = 2020
    month: int = 3
    day: int = 2
    hour: int = 1
    station: str = 'sas'
    experiment_id: int = -3503
    experiment_name: str = 'Normalscan'
    experiment_comment: str = 'fake comment for testing'
    freq: int = 10500
    scheduling_mode: str = 'discretionary'

    def __str__(self):
        """A pretty default print out format."""
        formatted_string = (
            f"{self.filename}\n"
            f"\tpath: {self.path}\n"
            f"\tversion: {self.version}\n"
            f"\tstructure: {self.structure}\n"
            f"\ttype: {self.type}\n"
            f"\tdatetime: {self.year:4d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:00\n"
            f"\tstation: {self.station}\n"
            f"\texperiment: {self.experiment_id} {self.experiment_name} '{self.experiment_comment}'\n"
            f"\tmode: {self.scheduling_mode}\n"
            f"\tfrequency: {self.freq}\n"
        )
        return formatted_string


def main(args):
    """
    Order of operations:
    1) Fetch the files. Checks if single file or directory and if it should work recursively. Also checks patterns.
    2) Query the files. If any Query is set then use them to filter the list. On single file can tell yes/no pass.
    3) Check if we want to return a list, meta, walk, or batch.
    4) If batch make the batch tuples.

    Parameters
    ----------
    args

    Returns
    -------

    """

    # file or dir -> recursive? -> check pattern -> check query -> return meta or walk or list?
    files = Fetch(args.filepath, args.pattern, args.recursive, args.walk)
    if args.queries:
        queries = args.query.split(',')
        files, files_data = Query(**dict(q.split('=') for q in queries))

    if args.batch:
        # Todo (adam):
        # do batch stuff
        # return batch tuple probably only for convert
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperDARN Canada© -- Engineering Diagnostic Tools Kit: '
                                                 '(Fetcher) '
                                                 'Given a search query (--query) or file name patter (--pattern) and '
                                                 'parent directory (--directory) returns a list of data files.'
                                                 'If a single file is given returns the basic meta data.')
    # parser.add_argument("filepath", help="input file or parent directory.")
    parser.add_argument('-p', '--pattern', type=str, help='the file naming pattern less the appending numbers.')
    parser.add_argument('-q', '--query', type=str, help='enter a search query to filter files, '
                                                        'eg: --query="station=sas,structure=array".')
    parser.add_argument('-r', '--recursive', action='store_true', help='recursively search if given a parent directory.')
    parser.add_argument('-v', '--verbose', action='store_true', help='explain what is being done verbosely.')
    parser.add_argument('-w', '--walk', action='store_true', help='walk the data file rather than print basic meta data.')
    parser.add_argument('-l', '--list', action='store_true', help='return a list of file paths rather than print basic meta data.')

    main(parser.parse_args())

    # f = ['/home/glatteis/SuperDARN/data/20210225.2219.22.sas.0.rawacf.hdf5.array',
    #      '/home/glatteis/SuperDARN/data/20210225.2219.22.sas.0.rawacf.hdf5.site']
    # dc = FileData()
    # print(dc)

