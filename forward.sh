export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f mpiconfig_gpu \
python fwi.py config/multiple.yml \
--mode forward \
--opt adam \
--use-cuda
# forward inversion