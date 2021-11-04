# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting antennas_iq files
to bfiq files.
"""
import itertools
import subprocess as sp
import math
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

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing,
                centre_offset=0.0):
    """
    Find the phase shift for a given antenna and beam direction.
    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
    between antennas.
    :param beamdir: the azimuthal direction of the beam off boresight, in degrees, positive beamdir being to
        the right of the boresight (looking along boresight from ground). This is for this antenna.
    :param freq: transmit frequency in kHz
    :param antenna: antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the boresight
        and positive beamdir right of boresight
    :param pulse_shift: in degrees, for phase encoding
    :param num_antennas: number of antennas in this array
    :param antenna_spacing: distance between antennas in this array, in meters
    :param centre_offset: the phase reference for the midpoint of the array. Default = 0.0, in metres.
     Important if there is a shift in centre point between arrays in the direction along the array.
     Positive is shifted to the right when looking along boresight (from the ground).
    :returns phshift: a phase shift for the samples for this antenna number, in radians.
    """
    freq = freq * 1000.0  # convert to Hz.

    # Convert to radians
    beamrad = math.pi * np.float64(beamdir) / 180.0

    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    phshift = 2 * math.pi * freq * \
              (((num_antennas - 1) / 2.0 - antenna) * antenna_spacing + centre_offset) * \
              math.cos(math.pi / 2.0 - beamrad) / speed_of_light

    phshift = phshift + math.radians(pulse_shift)

    # Bring into range (-2*pi, 2*pi)
    phshift = math.fmod(phshift, 2 * math.pi)

    return phshift


def shift_samples(basic_samples, phshift, amplitude):
    """
    Shift samples for a pulse by a given phase shift.
    Take the samples and shift by given phase shift in rads and adjust amplitude as
    required for imaging.
    :param basic_samples: samples for this pulse, numpy array
    :param phshift: phase for this antenna to offset by in rads, float
    :param amplitude: amplitude for this antenna (= 1 if not imaging), float
    :returns samples: basic_samples that have been shaped for the antenna for the
     desired beam.
    """
    # print(len(phshift), len(phshift[0]), len(basic_samples), len(basic_samples[0]))
    samples = amplitude * np.exp(1j * phshift) * basic_samples

    return samples


def beamform(antennas_data, beamdirs, rxfreq, antenna_spacing, pulse_phase_offset):
    """
    :param antennas_data: numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be
    from the same array and are assumed to be side by side with antenna spacing 15.24 m, pulse_shift = 0.0
    :param beamdirs: list of azimuthal beam directions in degrees off boresite
    :param rxfreq: frequency to beamform at.
    :param antenna_spacing: spacing in metres between antennas, used to get the phase shift that
    corresponds to an azimuthal direction.
    :param pulse_phase_offset: offset phase to adjust beams by. degrees
    """
    beamformed_data = []

    # [num_antennas, num_samps]
    num_antennas, num_samps = antennas_data.shape
    num_pulses = len(pulse_phase_offset)

    # Loop through all beam directions
    for beam_direction in beamdirs:
        antenna_phase_shifts = []

        # Get phase shift for each antenna
        for antenna in range(num_antennas):
            phase_shift = get_phshift(beam_direction,
                                      rxfreq,
                                      antenna,
                                      0.0,
                                      num_antennas,
                                      antenna_spacing)
            # Bring into range (-2*pi, 2*pi) and negate
            phase_shift = math.fmod((1 * phase_shift), 2 * math.pi)
            antenna_phase_shifts.append(phase_shift)

        # TODO: Figure out decoding phase here
        # Apply phase shift to data from respective antenna
        phased_antenna_data = [shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0) for i in
                               range(num_antennas)]
        phased_antenna_data = np.array(phased_antenna_data)

        # Sum across antennas to get beamformed data
        one_beam_data = np.sum(phased_antenna_data, axis=0)
        beamformed_data.append(one_beam_data)
    beamformed_data = np.array(beamformed_data)

    return beamformed_data


def beamform_samples(samples, beam_phases):
    """
    Beamform the samples for multiple beams simultaneously.

    :param      samples:           The filtered input samples.
    :type       samples:           ndarray [num_slices, num_antennas, num_samples]
    :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                   combining.
    :type       beam_phases:       list

    """
    beam_phases = xp.array(beam_phases)

    # samples:              [num_slices, num_antennas, num_samples]
    # beam_phases:          [num_slices, num_beams, num_antennas]
    # beamformed_samples:   [num_slices, num_beams, num_samples]
    beamformed_samples = xp.einsum('ijk,ilj->ilk', samples, beam_phases)

    return beamformed_samples


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
    intf_antenna_count = record['intf_antenna_count']

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
        """Converts lists stored in dict into numpy array. Recursive.
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
