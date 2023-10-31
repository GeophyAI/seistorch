mpirun -f hosts \
python ../../fwi.py forward.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed \
--use-cuda