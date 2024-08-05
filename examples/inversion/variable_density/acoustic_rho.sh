mpirun -f hosts \
python ../../../fwi.py configs/acoustic_rho.yml \
--mode forward \
--num-batches 1 \
--use-cuda