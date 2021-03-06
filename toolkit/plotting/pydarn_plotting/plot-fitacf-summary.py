
import sys
import pydarn
import matplotlib.pyplot as plt

fitacf_file = sys.argv[1]

fitacf_data = pydarn.SuperDARNRead.read_dmap(fitacf_file)

for beam in range(0,16):
    pydarn.RTP.plot_summary(fitacf_data, figsize=(22, 17), beam_num=beam, groundscatter=True)
    plt.savefig(fitacf_file + '.beam{}.summaryplot.png'.format(beam), dpi=400)
#plt.show()

