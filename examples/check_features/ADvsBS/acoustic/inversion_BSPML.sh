torchrun --nproc_per_node=1 ../../../../seistorch_dist.py configs/inversion_BSPML.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/fwi_classic_BSPML \
--use-cuda