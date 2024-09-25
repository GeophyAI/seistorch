mpirun -f hosts \
python ../../../forward.py configs/acoustic_habc_multiple.yml \
--mode forward \
--num-batches 1 \
--use-cuda