mpirun -f hosts \
python ../../../forward.py configs/acoustic.yml \
--mode forward \
--num-batches 1 \
--use-cuda