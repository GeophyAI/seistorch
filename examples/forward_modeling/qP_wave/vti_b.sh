mpirun -f hosts \
python ../../../forward.py configs/vti_b.yml \
--mode forward \
--num-batches 1 \
--use-cuda