mpirun -f hosts \
python ../../../../fwi.py inversion.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 2 \
--lr vp=20.0 \
--mode inversion \
--save-path ./results/towed_l2 \
--grad-cut \
--use-cuda