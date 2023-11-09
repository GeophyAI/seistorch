mpirun -f ../hosts \
python ../../../../../fwi.py forward_withmultiple.yml \
--mode forward \
--num-batches 10 \
--use-cuda