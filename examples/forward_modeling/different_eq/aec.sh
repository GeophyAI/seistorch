mpirun -f hosts \
python ../../../fwi.py configs/aec.yml \
--mode forward \
--num-batches 1 \
--use-cuda