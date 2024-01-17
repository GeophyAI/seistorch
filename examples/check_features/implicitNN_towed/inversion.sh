mpirun -f hosts \
python /home/shaowinw/Desktop/seistorch/fwi.py inversion.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 4 \
--lr vp=0.0001 \
--mode inversion \
--save-path ./results/towed \
--use-cuda