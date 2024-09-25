mpirun -f hosts \
python ../../../../seistorch/forward.py configs/forward_obs.yml \
--mode forward \
--num-batches 4 \
--use-cuda
