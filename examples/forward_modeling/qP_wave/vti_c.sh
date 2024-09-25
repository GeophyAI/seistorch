mpirun -f hosts \
python ../../../forward.py configs/vti_c.yml \
--mode forward \
--num-batches 1 \
--use-cuda