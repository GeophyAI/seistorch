export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../forward.py rtm.yml  \
--opt adam \
--loss vp=rtm \
--num-batches 30 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/rtm_true \
--use-cuda