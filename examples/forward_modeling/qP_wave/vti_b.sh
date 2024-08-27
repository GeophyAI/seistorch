mpirun -f hosts \
python ../../../fwi.py configs/vti_b.yml \
--mode forward \
--num-batches 1 \
--use-cuda