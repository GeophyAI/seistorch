mpirun -f hosts \
python /home/shaowinw/Desktop/seistorch/fwi.py forward.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 4 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed_traditional \
--grad-cut \
--use-cuda