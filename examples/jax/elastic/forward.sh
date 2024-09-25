mpirun -f hosts \
python ../../../forward.py configs/forward.yml \
--mode forward \
--num-batches 4 \
--use-cuda