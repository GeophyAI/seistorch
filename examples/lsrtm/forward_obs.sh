mpirun -f hosts \
python /home/shaowinw/seistorch/fwi.py forward_obs.yml \
--mode forward \
--num-batches 4 \
--use-cuda
