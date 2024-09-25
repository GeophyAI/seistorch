mpirun -f hosts \
python ../../../forward.py configs/acoustic_rho.yml \
--mode forward \
--num-batches 1 \
--use-cuda