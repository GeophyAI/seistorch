mpirun -f hosts \
python ../../../../fwi.py configs/forward_npy.yml \
--mode forward \
--num-batches 4 \
--use-cuda