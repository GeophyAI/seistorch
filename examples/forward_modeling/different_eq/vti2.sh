mpirun -f hosts \
python ../../../forward.py configs/vti2.yml \
--mode forward \
--num-batches 1 \
--use-cuda