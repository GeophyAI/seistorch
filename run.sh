export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
mpirun -f mpiconfig_gpu \
python fwi.py config/example_marmousi.yml mpiconfig_gpu \
--mode inversion \
--use-cuda
# forward inversion