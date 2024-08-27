mpirun -f hosts \
python ../../../fwi.py configs/vti_c.yml \
--mode forward \
--num-batches 1 \
--use-cuda