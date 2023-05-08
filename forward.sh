mpirun -f mpiconfig_gpu \
python fwi.py config/coding.yml mpiconfig_gpu \
--mode forward \
--opt adam \
--use-cuda
# forward inversion