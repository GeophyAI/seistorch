torchrun --nproc_per_node=1 ../../../../seistorch/seistorch_dist.py lsrtm.yml  \
--opt adam \
--loss rvp=l2 rvs=l2 \
--lr rvp=0.01 rvs=0.01 \
--mode inversion \
--save-path ./results \
--use-cuda \
--grad-cut