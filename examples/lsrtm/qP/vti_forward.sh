mpirun -f hosts \
python ../../../fwi.py configs/vti.yml \
--mode forward \
--num-batches 1 \
--use-cuda