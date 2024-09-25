mpirun -f hosts \
python ../../../forward.py configs/decouple.yml \
--mode forward \
--num-batches 1 \
--use-cuda