mpirun -f configs/hosts \
python ../../../../fwi.py configs/forward.yml \
--mode forward \
--use-cuda