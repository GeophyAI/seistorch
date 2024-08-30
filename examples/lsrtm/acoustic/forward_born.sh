mpirun -f hosts \
python ../../../../seistorch/fwi.py configs/forward_born.yml \
--mode forward \
--num-batches 4 \
--use-cuda
