torchrun --nproc_per_node=4 /home/shaowinw/seistorch/seistorch_dist.py configs/inversion_ADHABC.yml  \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/fwi_classic_ADHABC \
--use-cuda