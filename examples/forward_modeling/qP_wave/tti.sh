mpirun -f hosts \
python ../../../fwi.py configs/tti.yml \
--mode forward \
--num-batches 1 \
--use-cuda