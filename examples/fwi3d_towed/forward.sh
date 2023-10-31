mpirun -f hosts \
python ../../fwi.py forward.yml \
--mode forward \
--num-batches 4 \
--use-cuda