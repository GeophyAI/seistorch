mpirun -f hosts \
python ../../../../fwi.py configs/forward_hdf5.yml \
--mode forward \
--num-batches 4 \
--use-cuda