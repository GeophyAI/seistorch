export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../fwi.py config/nodatamask_forip.yml \
--opt steepestdescent \
--loss vp=ip \
--num-batches 1 \
--lr vp=30.0 \
--mode inversion \
--save-path ./results/ip \
--grad-cut \
--use-cuda