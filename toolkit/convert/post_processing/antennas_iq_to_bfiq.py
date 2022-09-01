"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (Data Converting & Post Processing)

Author(s): Marci Detwiller, Adam Lozinsky, Remington Rohel
Date: November 5, 2021
Affiliation: University of Saskatchewan

This file contains the antennas_iq_to_bfiq() function and associated helper functions. The primary purpose is
post processing of antennas_iq.site and antennas_iq.array borealis data files.
"""

import itertools
import subprocess as sp
import os
import numpy as np
import deepdish as dd
from scipy.constants import speed_of_light

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('convert')


def phase_shift(beam_direction, frequency, antenna, pulse_shift, num_antennas, antenna_spacing, center_offset=0.0):
    """
    Find the phase shift for a given antenna and beam direction.
    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
    between antennas.

    Parameters
    ----------
        beam_direction :
            The azimuthal direction of the beam off boresight, in degrees, positive beamdir being to the right of the
            boresight (looking along boresight from ground). This is for this antenna.
        frequency:
            Transmit frequency in kHz.
        antenna :
            Antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the boresight
            and positive beamdir right of boresight.
        pulse_shift :
            in degrees, for phase encoding.
        num_antennas :
            Number of antennas in this array.
        antenna_spacing :
            Distance between antennas in this array, in meters
        center_offset :
            The phase reference for the midpoint of the array. Default = 0.0, in metres. Important if there is a shift
            in centre point between arrays in the direction along the array. Positive is shifted to the right when
            looking along boresight (from the ground).

    Returns
    -------
        phase :
            A phase shift for the samples for this antenna number, in radians.
    """

    wavelength = speed_of_light / (frequency * 1000.0)
    separation = ((num_antennas - 1) / 2.0 - antenna) * antenna_spacing + center_offset
    phase = (2 * np.pi * separation / wavelength) * np.cos(np.pi / 2.0 - np.deg2rad(beam_direction))
    phase += np.rad2deg(pulse_shift)
    phase = np.fmod(phase, 2 * np.pi)

    return phase


def shift_samples(samples, phase, amplitude):
    """
    Shift samples for a pulse by a given phase shift. Take the samples and shift by given phase shift in radians
    then adjust amplitude as required for imaging.

    Parameters
    ----------
        samples : complex32 np.array
            Complex samples for this pulse.
        phase: float np.array
            Phase shift for specific antenna to offset by in radians.
        amplitude :
            amplitude for this antenna (= 1 if not imaging), float

    Returns
    -------
        samples : complex32 np.array
            Samples that have been shaped for the antenna for the desired beam.
    """
    print(phase.shape)
    samples *= amplitude * np.exp(1j * phase)

    return samples


def beamform(antennas_data, beam_directions, frequency, antenna_spacing, pulse_phase_offset=0.0):
    """
    Given complex antenna data from the linear SuperDARN array, either main or intf, beamfroms the data in the
    direction given and adjusts for pulse shift.

    Parameters
    ----------
        antennas_data : complex64 np.array
            [num_antennas, num_samples] All antennas are assumed to be from the same array and to be side by side with
            antenna spacing 15.24 m, pulse_shift = 0.0.
        beam_directions : float np.array
            Azimuthal beam directions in degrees off boresight.
        frequency : float
            Receive frequency to beamform at.
        antenna_spacing : float
            Spacing in metres between antennas, used to get the phase shift that corresponds to an azimuthal direction.
        pulse_shift : float
            Offset phase in degrees to adjust the beams by.

    Returns
    -------
        beamformed_data : complex64 np.array
            The data beam formed in the direction given.
    """

    beamformed_data = np.array([])
    num_antennas, num_samps = antennas_data.shape

    # Loop through all beam directions
    for beam_direction in beam_directions:
        antenna_phase_shifts = \
            phase_shift(beam_direction, frequency, np.arange(num_antennas), pulse_phase_offset, num_antennas, antenna_spacing)
        antenna_phase_shifts = np.repeat(antenna_phase_shifts[:, np.newaxis], num_samps, axis=1)
        # TODO: Figure out decoding phase here
        phased_antenna_data = shift_samples(antennas_data, antenna_phase_shifts, 1.0)
        beamformed_data = np.append(beamformed_data, np.sum(phased_antenna_data, axis=0))

    return beamformed_data


def beamform_record2(filename, out_file):
    def check_dataset_add(k, v):
        if k not in recs[group_name].keys():
            recs[group_name][k] = v
            if key_num == 0:
                print(f'\t- added: {k}')

    def check_dataset_rename(k, v):
        if k in recs[group_name].keys():
            recs[group_name][v] = recs[group_name][k]
            del recs[group_name][k]
            if key_num == 0:
                print(f'\t- updated: {k}')

    def check_dataset_del(k):
        if k in recs[group_name].keys():
            del recs[group_name][k]
            if key_num == 0:
                print(f'\t- removed: {k}')

                if 'timestamp_of_write' in recs[group_name].keys():
                    del recs[group_name]['timestamp_of_write']
                    if key_num == 0:
                        print('timestamp_of_write removed')

    def check_dataset_revalue(k, v):
        if k in recs[group_name].keys():
            recs[group_name][k] = v
            if key_num == 0:
                print(f'\t- updated: {k}')

    # Update the file
    print(f'file: {filename}')

    for key_num, group_name in enumerate(sorted_keys):
        # Find all the bfiq required missing datasets or create them

        # first_range
        first_range = 180.0  #scf.FIRST_RANGE
        check_dataset_add('first_range', np.float32(first_range))

        # first_range_rtt
        first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
        check_dataset_add('first_range_rtt', np.float32(first_range_rtt))

        # lags
        lag_table = list(itertools.combinations(recs[group_name]['pulses'], 2))
        lag_table.append([recs[group_name]['pulses'][0], recs[group_name]['pulses'][0]])  # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
        lag_table.append([recs[group_name]['pulses'][-1], recs[group_name]['pulses'][-1]])  # alternate lag 0
        lags = np.array(lag_table, dtype=np.uint32)
        check_dataset_add('lags', lags)

        # num_ranges
        if station_name in ["cly", "rkn", "inv"]:
            num_ranges = 100  # scf.POLARDARN_NUM_RANGES
            check_dataset_add('num_ranges', np.uint32(num_ranges))
        elif station_name in ["sas", "pgr"]:
            num_ranges = 75  # scf.STD_NUM_RANGES
            check_dataset_add('num_ranges', np.uint32(num_ranges))

        # range_sep
        range_sep = 1 / recs[group_name]['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0
        check_dataset_add('range_sep', np.float32(range_sep))

        # Check pulse_phase_offset type
        recs[group_name]['pulse_phase_offset'] = np.float32(recs[group_name]['pulse_phase_offset'][()])

        # Beamform the data
        main_antenna_spacing = 15.24  # For SAS from config file
        intf_antenna_spacing = 15.24  # For SAS from config file
        beam_azms = recs[group_name]['beam_azms'][()]
        freq = recs[group_name]['freq']

        # antennas data shape  = [num_antennas, num_sequences, num_samps]
        antennas_data = recs[group_name]['data']
        antennas_data = antennas_data.reshape(recs[group_name]['data_dimensions'])
        main_beamformed_data = np.array([], dtype=np.complex64)
        intf_beamformed_data = np.array([], dtype=np.complex64)

        # Loop through every sequence and beamform the data
        # output shape after loop is [num_sequences, num_beams, num_samps]
        for j in range(antennas_data.shape[1]):
            # data input shape = [num_antennas, num_samps]
            # data return shape = [num_beams, num_samps]
            main_beamformed_data = np.append(main_beamformed_data,
                                             beamform(antennas_data[0:16, j, :], beam_azms, freq, main_antenna_spacing))
            intf_beamformed_data = np.append(intf_beamformed_data,
                                             beamform(antennas_data[16:20, j, :], beam_azms, freq, intf_antenna_spacing))

        # Remove iq data for bfiq data.
        # Data shape after append is [num_antenna_arrays, num_sequences, num_beams, num_samps]
        # Then flatten the array for final .site format
        del recs[group_name]['data']
        recs[group_name]['data'] = np.append(main_beamformed_data, intf_beamformed_data).flatten()

        # data_dimensions
        # We need set num_antennas_arrays=2 for two arrays and num_beams=length of beam_azms
        data_dimensions = recs[group_name]['data_dimensions']
        recs[group_name]['data_dimensions'] = np.array([2, data_dimensions[1], len(beam_azms), data_dimensions[2]], dtype=np.uint32)

        # NOTE (Adam): After all this we essentially could loop through all records and build the array file live but,
        # it is just as easy to save the .site format and use pydarnio to reload the data, verify its contents
        # automatically and then reshape it into .array format (which automatically handles all the zero padding).

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file + '.tmp', dtstr=group_name)

        sp.call(cmd.split())
        os.remove(tmp_file)

    bfiq_reader = pydarnio.BorealisRead(out_file + '.tmp', 'bfiq', 'site')
    array_data = bfiq_reader.arrays
    bfiq_writer = pydarnio.BorealisWrite(out_file, array_data, 'bfiq', 'array')

    os.remove(out_file + '.tmp')
    print('out_file:', out_file)

    return


def beamform_record(record):
    """
    Takes a record from an antennas_iq file and beamforms the data.

    :param record:      Borealis antennas_iq record
    :return:            Record of beamformed data for bfiq site file
    """

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------ First Range --------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    first_range = 180.0  # scf.FIRST_RANGE
    record['first_range'] = np.float32(first_range)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- First Range Round Trip Time ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
    record['first_range_rtt'] = np.float32(first_range_rtt)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Create Lag Table ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    lag_table = list(itertools.combinations(record['pulses'], 2))  # Create all combinations of lags
    lag_table.append([record['pulses'][0], record['pulses'][0]])  # lag 0
    lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
    lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
    lags = np.array(lag_table, dtype=np.uint32)
    record['lags'] = lags

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Number of Ranges ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TODO: Do this intelligently. Maybe grab from githash and cpid? Have default values too

    station = record['station']
    if station in ["cly", "rkn", "inv"]:
        num_ranges = 100  # scf.POLARDARN_NUM_RANGES
        record['num_ranges'] = np.uint32(num_ranges)
    elif station in ["sas", "pgr"]:
        num_ranges = 75  # scf.STD_NUM_RANGES
        record['num_ranges'] = np.uint32(num_ranges)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Range Separation ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    range_sep = 1 / record['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0
    record['range_sep'] = np.float32(range_sep)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Beamform the data ----------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    beam_azms = record['beam_azms']
    freq = record['freq']
    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    # antennas data shape  = [num_antennas, num_sequences, num_samps]
    antennas_data = record['data']

    # Get the data and reshape
    num_antennas, num_sequences, num_samps = record['data_dimensions']
    antennas_data = antennas_data.reshape(record['data_dimensions'])

    main_beamformed_data = np.array([], dtype=np.complex64)
    intf_beamformed_data = np.array([], dtype=np.complex64)
    main_antenna_count = record['main_antenna_count']

    # TODO: Grab these values from somewhere
    main_antenna_spacing = 15.24
    intf_antenna_spacing = 15.24

    # Loop through every sequence and beamform the data.
    # Output shape after loop is [num_sequences, num_beams, num_samps]
    for sequence in range(num_sequences):
        # data input shape  = [num_antennas, num_samps]
        # data return shape = [num_beams, num_samps]
        main_beamformed_data = np.append(main_beamformed_data,
                                         beamform(antennas_data[:main_antenna_count, sequence, :],
                                                  beam_azms,
                                                  freq,
                                                  main_antenna_spacing,
                                                  pulse_phase_offset))
        intf_beamformed_data = np.append(intf_beamformed_data,
                                         beamform(antennas_data[main_antenna_count:, sequence, :],
                                                  beam_azms,
                                                  freq,
                                                  intf_antenna_spacing,
                                                  pulse_phase_offset))

    record['data'] = np.append(main_beamformed_data, intf_beamformed_data).flatten()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- Data Descriptors & Dimensions ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Old dimensions: [num_antennas, num_sequences, num_samps]
    # New dimensions: [num_antenna_arrays, num_sequences, num_beams, num_samps]
    data_descriptors = record['data_descriptors']
    record['data_descriptors'] = ['num_antenna_arrays',
                                  data_descriptors[1],
                                  'num_beams',
                                  data_descriptors[2]]
    record['data_dimensions'] = np.array([2, num_sequences, len(beam_azms), num_samps],
                                         dtype=np.uint32)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- Antennas Array Order ---------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['antenna_arrays_order'] = ['main', 'intf']

    return record


def antennas_iq_to_bfiq(infile, outfile):
    """
    Converts an antennas_iq site file to bfiq site file

    :param infile:      Borealis antennas_iq site file
    :type  infile:      String
    :param outfile:     Borealis bfiq site file
    :type  outfile:     String
    :return:            Path to bfiq site file
    """

    def convert_to_numpy(data):
        """
        Converts lists stored in dict into numpy array. Recursive.
        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        return data

    postprocessing_logger.info('Converting file {} to bfiq'.format(infile))

    # Load file to read in records
    bfiq_group = dd.io.load(infile)
    records = bfiq_group.keys()

    # Convert each record to bfiq record
    for record in records:
        beamformed_record = beamform_record(bfiq_group[record])

        # Convert to numpy arrays for saving to file with deepdish
        formatted_record = convert_to_numpy(beamformed_record)

        # Save record to temporary file
        tempfile = '/tmp/{}.tmp'.format(record)
        dd.io.save(tempfile, formatted_record, compression=None)

        # Copy record to output file
        cmd = 'h5copy -i {} -o {} -s {} -d {}'
        cmd = cmd.format(tempfile, outfile, '/', '/{}'.format(record))
        sp.call(cmd.split())

        # Remove temporary file
        os.remove(tempfile)
