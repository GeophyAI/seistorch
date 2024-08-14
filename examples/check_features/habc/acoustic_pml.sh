mpirun -f hosts \
python ../../../fwi.py configs/acoustic_pml.yml \
--mode forward \
--num-batches 1 \
--use-cuda