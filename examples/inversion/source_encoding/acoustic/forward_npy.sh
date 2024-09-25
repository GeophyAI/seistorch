mpirun -f hosts \
python ../../../../forward.py configs/forward_npy.yml \
--mode forward \
--num-batches 4 \
--use-cuda