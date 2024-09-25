mpirun -f hosts \
python ../../../../forward.py configs/forward_hdf5.yml \
--mode forward \
--num-batches 10 \
--use-cuda