mpirun -f hosts \
python ../../../fwi.py forward.yml \
--mode forward \
--num-batches 10 \
--use-cuda