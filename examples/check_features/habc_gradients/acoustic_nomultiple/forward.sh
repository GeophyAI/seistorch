mpirun -f configs/hosts \
python /root/seistorch/forward.py configs/forward.yml \
--mode forward \
--use-cuda