mpirun -f hosts \
python ../../../../forward.py configs/initial.yml \
--mode forward \
--num-batches 1 \
--use-cuda