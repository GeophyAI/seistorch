torchrun --nproc_per_node=1 /root/seistorch/seistorch_dist.py configs/inversion_BSHABC.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/fwi_classic_BSHABC \
--use-cuda