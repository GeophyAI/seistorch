export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../fwi.py forward.yml \
--opt steepestdescent \
--loss vp=w1d \
--num-batches 1 \
--lr vp=20.0 \
--mode inversion \
--save-path ./results_w1d \
--use-cuda