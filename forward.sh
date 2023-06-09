# export PATH=/usr/local/mpich/bin:$PATH && \
# export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f mpiconfig_gpu \
python fwi.py config/coding_elastic.yml mpiconfig_gpu \
--mode forward \
--opt adam \
--use-cuda
# forward inversion