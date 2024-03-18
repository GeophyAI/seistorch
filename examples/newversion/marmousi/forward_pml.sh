mpirun -f hosts \
python /home/shaowinw/seistorch/fwi.py config/pml.yml \
--mode forward \
--num-batches 4 \
--use-cuda