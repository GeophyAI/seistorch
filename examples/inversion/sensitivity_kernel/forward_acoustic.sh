mpirun -f hosts \
python ../../../fwi.py configs/forward_acoustic_habc.yml \
--mode forward \
--num-batches 1 \
--use-cuda