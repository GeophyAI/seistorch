export PATH=/root/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/root/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../fwi.py configs/acoustic.yml \
--mode forward \
--num-batches 1 \
--use-cuda