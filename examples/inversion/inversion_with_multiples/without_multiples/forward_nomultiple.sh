mpirun -f ../hosts \
python ../../../../../forward.py forward_nomultiple.yml \
--mode forward \
--num-batches 10 \
--use-cuda