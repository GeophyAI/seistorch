mpirun -f hosts \
python ../../../forward.py configs/aec.yml \
--mode forward \
--num-batches 1 \
--use-cuda