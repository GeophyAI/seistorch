mpirun -f hosts \
python ../../../fwi.py configs/acoustic_habc_multiple.yml \
--mode forward \
--num-batches 1 \
--use-cuda