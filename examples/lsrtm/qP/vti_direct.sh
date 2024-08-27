mpirun -f hosts \
python ../../../fwi.py configs/vti_direct_wave.yml \
--mode forward \
--num-batches 1 \
--use-cuda