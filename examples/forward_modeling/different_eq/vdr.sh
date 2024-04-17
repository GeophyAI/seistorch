mpirun -f hosts \
python ../../../fwi.py configs/vdr.yml \
--mode forward \
--num-batches 1 \
--use-cuda