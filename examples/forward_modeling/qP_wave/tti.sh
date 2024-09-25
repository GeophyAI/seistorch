mpirun -f hosts \
python ../../../forward.py configs/tti.yml \
--mode forward \
--num-batches 1 \
--use-cuda