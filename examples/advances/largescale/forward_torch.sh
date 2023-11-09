torchrun --nproc_per_node=4 ../../fwi_torchrun.py forward.yml \
--mode forward \
--modelparallel \
--num-batches 4 \
--use-cuda