mpirun -f hosts \
python ../../../forward.py forward.yml \
--mode forward \
--num-batches 1 \
--use-cuda