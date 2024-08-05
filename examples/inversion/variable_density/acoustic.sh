mpirun -f hosts \
python ../../../fwi.py configs/acoustic.yml \
--mode forward \
--num-batches 1 \
--use-cuda