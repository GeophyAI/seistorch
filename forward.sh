export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f mpiconfig_gpu \
python fwi.py /public1/home/wangsw/FWI/NO_LOWFREQ/config/00_generate_p_by_acoustic_init.yml \
--mode forward \
--use-cuda
# forward inversion