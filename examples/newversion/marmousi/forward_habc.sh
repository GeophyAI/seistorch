mpirun -f hosts \
python /home/shaowinw/seistorch/fwi.py config/habc.yml \
--mode forward \
--num-batches 4 \
--use-cuda