mpirun -f hosts \
python ../../../fwi.py vdr_bg.yml \
--mode forward \
--num-batches 5 \
--use-cuda