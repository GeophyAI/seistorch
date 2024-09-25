mpirun -f hosts \
python ../../../forward.py configs/tti_born.yml \
--mode forward \
--num-batches 1 \
--use-cuda