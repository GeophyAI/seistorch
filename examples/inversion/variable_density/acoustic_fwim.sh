mpirun -f hosts \
python ../../../forward.py configs/acoustic_fwim.yml \
--mode forward \
--num-batches 1 \
--use-cuda