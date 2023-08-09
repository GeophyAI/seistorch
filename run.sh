export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
mpirun -n 2 \
python fwi.py config/check/layer2d.yml  \
--opt adam \
--loss l2 \
--lr vp=10.0 \
--mode inversion \
--save-path /home/wangsw/inversion/2d/layer/results/l2_nocoding \
--use-cuda