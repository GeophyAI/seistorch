mpirun -f hosts \
python ../../../forward.py configs/vdr_bg.yml \
--mode forward \
--num-batches 5 \
--use-cuda