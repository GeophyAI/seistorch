mpirun -f hosts \
python ../../../forward.py configs/forward.yml \
--mode forward \
--num-batches 2 \
--use-cuda