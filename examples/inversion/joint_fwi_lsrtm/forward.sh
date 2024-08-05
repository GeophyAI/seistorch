mpirun -f hosts \
python ../../../fwi.py configs/forward.yml \
--mode forward \
--num-batches 2 \
--use-cuda