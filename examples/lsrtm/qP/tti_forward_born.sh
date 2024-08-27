mpirun -f hosts \
python ../../../fwi.py configs/tti_born.yml \
--mode forward \
--num-batches 1 \
--use-cuda