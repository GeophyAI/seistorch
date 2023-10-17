export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../fwi.py rtm_initmodel.yml  \
--opt adam \
--loss vp=rtm \
--num-batches 87 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/rtm_init \
--use-cuda