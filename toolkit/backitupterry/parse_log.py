"""
Backitupterry is used to parse backup logs and should probably be deleted!
"""
import os
import sys

files = ["/home/arl203/superdarn/travel/2022_pgr_cly/data/pgr_2022_backup_drive1a.log",
         "/home/arl203/superdarn/travel/2022_pgr_cly/data/pgr_2022_backup_drive1b.log",
         "/home/arl203/superdarn/travel/2022_pgr_cly/data/pgr_2022_backup_drive2a.log"]


def main():
    with open('pgr_2022_backup.log', 'a') as w:
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line[0] == '2':
                        print(line[0:-1])
                        w.write(line)
                f.close()
        w.close()


if __name__ == '__main__':
    main()
