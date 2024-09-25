mpirun -f configs/hosts \
python ../../../forward.py configs/forward.yml \
--mode forward \
--use-cuda