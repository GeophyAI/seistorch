mpirun -f hosts \
python ../../../../fwi.py configs/forward_with_true_vel.yml \
--mode forward \
--num-batches 1 \
--use-cuda