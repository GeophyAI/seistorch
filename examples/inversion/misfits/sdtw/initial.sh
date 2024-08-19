mpirun -f hosts \
python ../../../../fwi.py configs/initial.yml \
--mode forward \
--num-batches 1 \
--use-cuda