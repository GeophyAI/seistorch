mpirun -f hosts \
python /home/shaowinw/seistorch/forward.py config/habc.yml \
--mode forward \
--num-batches 4 \
--use-cuda