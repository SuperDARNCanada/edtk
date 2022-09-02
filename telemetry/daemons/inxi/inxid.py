import subprocess
import json
import datetime


config = {
    'path': '/home/arl203/superdarn/code/edtk',    # Path to write the json file
    'site': 'campus',           # Options: campus, lab, sas, cly, pgr, rkn, inv, bakker, prelate
    'name': 'sd-eng1',          # Device human-readable name
    'device': 'workstation',    # Options: workstation, main, cd, dds, nas, server
}


class DeviceInfo:
    def __init__(self, config):
        self.file = f"{config['path']}/{config['site']}_{config['name']}_inxi.json"
        self.site = config['site']
        self.name = config['name']
        self.device = config['device']
        self.run_command()

    @staticmethod
    def execute_cmd(cmd):
        # try/except block lets install script continue even if something fails
        try:
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as err:
            output = err.output
        return output.decode('utf-8')

    def run_command(self):
        # the --tty command is required since we are running inxi from within python
        cmd = f"inxi --tty --no-sudo -v 8 -w --weather-unit m --output json --output-file print"
        msg = self.execute_cmd(cmd)
        msg = json.loads(msg)
        msg['time'] = datetime.datetime.utcnow().strftime('YY-mm-DD HH:MM:SS')
        msg['site'] = self.site
        msg['name'] = self.name

        with open(self.file, 'w') as fp:
            json.dump(msg, fp)


if __name__ == '__main__':
    DeviceInfo(config)