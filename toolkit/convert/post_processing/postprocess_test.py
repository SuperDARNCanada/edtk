import numpy as np
import pydarnio
from toolkit.convert.post_processing import antennas_iq_to_bfiq as ab


# test_file = '/home/glatteis/SuperDARN/data/post_process_sample/20211006.2132.00.sas.0.antennas_iq.hdf5.site'
# compare_file = '/home/glatteis/SuperDARN/data/post_process_sample/20211006.2132.00.sas.0.bfiq.hdf5.site'
test_file = 'F://superdarn/data/sample_data/20211006.2132.00.sas.0.antennas_iq.hdf5.site'
compare_file = 'F://superdarn/data/sample_data//20211006.2132.00.sas.0.bfiq.hdf5.site'


test_data = pydarnio.BorealisRead(test_file, 'antennas_iq', 'site').records
compare_data = pydarnio.BorealisRead(compare_file, 'bfiq', 'site').records
print(f'Data read in:\n\t-input -> {test_file}\n\t-compare -> {compare_file}')

key = list(test_data.keys())[0]

cd = compare_data[key]['data']
td = ab.beamform_record(test_data[key])
td = td['data']

print(td.shape, cd.shape)
print(f'test: {td}')
print(f'compare: {cd}')
print(f'close? {np.allclose(td, cd)} {td[0]} {cd[0]}')

