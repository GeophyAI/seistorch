mpirun -f hosts \
python ~/Desktop/seistorch/fwi.py forward.yml \
--mode forward \
--use-cuda