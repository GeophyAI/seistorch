mpirun -f hosts \
python ../../../forward.py configs/vdr.yml \
--mode forward \
--num-batches 1 \
--use-cuda