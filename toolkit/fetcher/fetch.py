# Fetch the files and return a nice list maybe walk the files if requested.

import os
import glob

class Fetch:
    def __init__(self):
        self.file_list = []
        return self.file_list

    @staticmethod
    def fetch_files(filepath, pattern='', recursive=False):
        """
        Parameters
        ----------
            filepath : str
                The file path or parent directory to fetch files and data from.
            pattern : str
                The file naming pattern of files to load; eg. '*.array*' fetches all array files.
            recursive : boolean
                Determine whether the files should be grabbed recursively from the parent directory.

        Returns
        -------
        """
        files = []
        if os.path.isfile(filepath):
            files = [filepath]
        elif os.path.isdir(filepath):
            files = glob.glob(filepath + '/**/' + pattern, recursive=recursive)
        else:
            print(f"Input {filepath} does not exist; must be a valid file or directory.")
            exit()

        if not files:
            files = glob.glob(filepath + pattern, recursive=recursive)
            if not files:
                print(f'no files found at: {filepath} or no files found with pattern: {pattern}')
                exit()

        return files

    def meta_files(self):
        return

    @staticmethod
    def walk_files(self):
        return



