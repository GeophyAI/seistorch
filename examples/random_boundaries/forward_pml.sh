mpirun -f hosts \
python ../../fwi.py pml.yml \
--mode forward \
--num-batches 1 \
--use-cuda