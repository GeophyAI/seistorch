mpirun -f hosts \
python ../../../fwi.py elastic.yml \
--mode forward \
--num-batches 5 \
--use-cuda