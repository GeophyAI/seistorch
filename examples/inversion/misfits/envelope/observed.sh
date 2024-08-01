mpirun -f hosts \
python ../../../../fwi.py configs/observed.yml \
--mode forward \
--num-batches 1 \
--use-cuda