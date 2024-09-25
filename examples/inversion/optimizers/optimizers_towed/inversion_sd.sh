export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../forward.py forward.yml  \
--opt steepestdescent \
--loss vp=l2 \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed_sd \
--grad-cut \
--use-cuda