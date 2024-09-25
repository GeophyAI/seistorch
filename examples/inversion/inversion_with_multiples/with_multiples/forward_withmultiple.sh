mpirun -f ../hosts \
python ../../../../../forward.py forward_withmultiple.yml \
--mode forward \
--num-batches 10 \
--use-cuda