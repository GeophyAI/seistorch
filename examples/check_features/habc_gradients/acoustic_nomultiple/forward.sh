mpirun -f configs/hosts \
python /root/seistorch/fwi.py configs/forward.yml \
--mode forward \
--use-cuda