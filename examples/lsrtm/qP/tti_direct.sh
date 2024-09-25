mpirun -f hosts \
python ../../../forward.py configs/tti_direct_wave.yml \
--mode forward \
--num-batches 1 \
--use-cuda