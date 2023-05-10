mpirun -n 2 \
python fwi.py config/coding.yml config/coding.yml \
--opt adam \
--loss ncc \
--mode inversion \
--use-cuda