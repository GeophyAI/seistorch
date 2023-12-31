export PATH=/usr/local/mpich/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -f hosts \
python ../../../../fwi.py configs/without_low_frequency_ttcc.yml  \
--opt steepestdescent \
--loss vp=traveltime \
--num-batches 2 \
--lr vp=20.0 \
--mode inversion \
--save-path ./results/towed_withoutlow_ttcc \
--grad-cut \
--use-cuda