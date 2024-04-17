mpirun -f hosts \
python ../../../fwi.py configs/vti2.yml \
--mode forward \
--num-batches 1 \
--use-cuda