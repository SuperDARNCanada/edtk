import glob
import os
import importlib


def main():
    tools = glob.glob("toolkit/**/[!_]*.py", recursive=True)
    for tool in tools:
        module = importlib.import_module(tool[:-3].replace('/', '.'))
        print(f'{os.path.basename(tool)}\n'
              f'------------------------------------------------------------------------------------------'
              f'{module.__doc__}\n')


if __name__ == '__main__':
    main()
