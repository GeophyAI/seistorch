mpirun -f hosts \
python ../../../fwi.py configs/decouple.yml \
--mode forward \
--num-batches 1 \
--use-cuda