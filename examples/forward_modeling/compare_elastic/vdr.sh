mpirun -f hosts \
python ../../../fwi.py configs/vdr.yml \
--mode forward \
--num-batches 5 \
--use-cuda