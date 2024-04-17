import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shaowinw/seistorch')
from seistorch.show import SeisShow
import lesio
show = SeisShow()

rank = 0
rand_no = 0
dt = 0.001
freqs = [3, 6]

rootpath = r'/home/shaowinw/seistorch/examples/inversion/misfits/travel_time_misfit/results/traveltime_l2'

obs=np.load(f'{rootpath}/obs{rank}.npy')
syn=np.load(f'{rootpath}/syn{rank}.npy')
# adj=np.load(f'{rootpath}/adj{rank}.npy')

# obs = lesio.tools.filter(obs, dt=dt, forder=3, freqs=freqs, mode='bandpass')
# syn = lesio.tools.filter(syn, dt=dt, forder=3, freqs=freqs, mode='bandpass')

amp_obs, freqs = lesio.tools.freq_spectrum(obs[rand_no], dt)
amp_syn, freqs = lesio.tools.freq_spectrum(syn[rand_no], dt)
amp_obs = amp_obs[freqs<10.]
amp_syn = amp_syn[freqs<10.]

freqs = freqs[freqs<10.]
plt.plot(freqs, amp_obs, label='obs')
plt.plot(freqs, amp_syn, label='syn')
plt.legend()
plt.show()

show.alternate(obs[rand_no], syn[rand_no], interval=20, )

show.shotgather([obs[rand_no], syn[rand_no]], 
                ['obs', 'syn'], 
                inarow=True, 
                dx = 25, 
                dt = dt,
                normalize=False, 
                show=True,
                cmap='seismic',
                savepath=None,)

show.wiggle([obs[rand_no], syn[rand_no]],
            colorlist=['r', 'b'],
            labellist=['obs', 'syn'],
            dx = 25, 
            dt = dt,
            downsample= 32, 
            show=True,
            savepath=None,)