import numpy as np
import pydarnio
from toolkit.postprocess import antennas_iq_to_bfiq as ab


test_file = '/home/glatteis/SuperDARN/data/post_process_sample/20211006.2132.00.sas.0.antennas_iq.hdf5.site'
compare_file = '/home/glatteis/SuperDARN/data/post_process_sample/20211006.2132.00.sas.0.bfiq.hdf5.site'

test_data = pydarnio.BorealisRead(test_file, 'antennas_iq', 'site').records
compare_data = pydarnio.BorealisRead(compare_file, 'bfiq', 'site').records
print(f'Data read in:\n\t-input -> {test_file}\n\t-compare -> {compare_file}')

key = list(test_data.keys())[0]
freq = test_data[key]['freq']
beam_azms = test_data[key]['beam_azms']
spacing = 15.24

print(test_data[key].keys())
# td = test_data[key]['data']
# td = td.reshape(test_data[key]['data_dimensions'])
cd = compare_data[key]['data']
# print(td.shape, cd.shape, test_data[key]['num_samps'])

td = ab.beamform_record(test_data[key])
# main = ab.beamform(td, beam_azms, freq, spacing, 0.0)
# intf = ab.beamform(td, beam_azms, freq, spacing, 0.0)
# beam_phases = np.ones((3,1,20), dtype=np.complex64)
# td = np.einsum('ijk,ilj->ilk', td, beam_phases)

td = td['data']
# print(td.shape, cd.shape)
# print(f'test: {td}')
# print(f'compare: {cd}')
print(f'close? {np.allclose(td, cd)} {td[0]} {cd[0]}')
# td = td * np.repeat(np.exp(1j), td.shape)
td = np.conj(td)
print(f'close fix? {np.allclose(td, cd)} {td[0]} {cd[0]}')

