mpirun -f ../hosts \
python ../../../../../fwi.py forward_nomultiple.yml \
--mode forward \
--num-batches 10 \
--use-cuda