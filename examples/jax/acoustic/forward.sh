export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
mpirun -f hosts \
python ../../../fwi.py configs/forward.yml \
--mode forward \
--num-batches 4 \
--use-cuda