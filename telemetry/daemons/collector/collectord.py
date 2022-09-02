"""
Collects data and parses into a JSON to send database for dashboard

The collector is the manager of all other telemetry daemons. It hadnles scheduling
when other daemons should run and makes pull requests from them for data. Some
daemons may need to run in a continuously monitoring mode, in this case the
collector listens for new data. The collector also packages all the telemetry data
into a JSON format file and pushes that to the dashboard database.

Dataclasses are preferred but JSON to DICT is so straight forward.

How should config be done? config.toml? config.py?
"""


def listen():

    return


def request():

    return


def schedule():

    return


def package_telemetry():

    return

