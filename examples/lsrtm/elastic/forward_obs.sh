mpirun -f hosts \
python ../../../../seistorch/fwi.py configs/forward_obs.yml \
--mode forward \
--num-batches 4 \
--use-cuda
