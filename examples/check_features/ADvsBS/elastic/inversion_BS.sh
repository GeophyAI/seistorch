export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f configs/hosts \
python ../../../../fwi.py configs/inversion_BS.yml  \
--opt adam \
--loss vp=l2 vs=l2 \
--num-batches 1 \
--lr vp=10.0 vs=5.78 \
--mode inversion \
--save-path ./results/fwi_classic_BS \
--use-cuda