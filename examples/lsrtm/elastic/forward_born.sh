mpirun -f hosts \
python ../../../../seistorch/forward.py configs/forward_born.yml \
--mode forward \
--num-batches 4 \
--use-cuda
