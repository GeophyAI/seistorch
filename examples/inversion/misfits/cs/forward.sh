mpirun -f hosts \
python ../../../../fwi.py configs/forward.yml \
--mode forward \
--num-batches 10 \
--use-cuda