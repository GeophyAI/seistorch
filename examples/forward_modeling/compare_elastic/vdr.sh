mpirun -f hosts \
python ../../../fwi.py vdr.yml \
--mode forward \
--num-batches 5 \
--use-cuda