export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../fwi.py forward.yml  \
--opt adam \
--loss vp=envelope \
--num-batches 4 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/envelope \
--grad-cut \
--use-cuda