mpirun -f mpiconfig_gpu \
python fwi.py config/viscoacoustic.yml mpiconfig_gpu \
--mode forward \
--opt adam \
--use-cuda
# forward inversion