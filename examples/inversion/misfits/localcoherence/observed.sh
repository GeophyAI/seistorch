mpirun -f hosts \
python ../../../../forward.py configs/observed.yml \
--mode forward \
--num-batches 1 \
--use-cuda