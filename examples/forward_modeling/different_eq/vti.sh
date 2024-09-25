mpirun -f hosts \
python ../../../forward.py configs/vti.yml \
--mode forward \
--num-batches 1 \
--use-cuda