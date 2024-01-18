mpirun -f hosts \
python /home/shaowinw/Desktop/seistorch/fwi.py configs/forward.yml \
--mode forward \
--num-batches 4 \
--use-cuda