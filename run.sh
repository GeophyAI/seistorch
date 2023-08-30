export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f mpiconfig_gpu \
python fwi.py /public1/home/wangsw/FWI/NO_LOWFREQ/config/02_acoustic_invert_p.yml  \
--opt adam \
--loss vp=fa \
--lr vp=10.0 \
--mode inversion \
--save-path /public1/home/wangsw/FWI/NO_LOWFREQ/fa2 \
--use-cuda \
--grad-cut