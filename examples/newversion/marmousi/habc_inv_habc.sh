torchrun --nproc_per_node=1 /home/shaowinw/seistorch/seistorch_dist.py config/inv_habc.yml  \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/habc_inv_habc \
--grad-cut \
--filteratfirst \
--use-cuda