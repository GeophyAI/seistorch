mpirun -f hosts \
python ../../fwi.py rand.yml \
--mode forward \
--num-batches 1 \
--use-cuda