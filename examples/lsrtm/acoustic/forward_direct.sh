mpirun -f hosts \
python ../../../../seistorch/fwi.py configs/forward_direct.yml \
--mode forward \
--num-batches 4 \
--use-cuda
