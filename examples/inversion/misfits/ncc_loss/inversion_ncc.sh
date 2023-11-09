mpirun -f hosts \
python ../../../../fwi.py inversion.yml  \
--opt adam \
--loss vp=cs \
--num-batches 2 \
--lr vp=20.0 \
--mode inversion \
--save-path ./results/towed_cs \
--grad-cut \
--use-cuda