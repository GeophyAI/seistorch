torchrun --nproc_per_node=1 ../../../../seistorch/seistorch_dist.py configs/lsrtm.yml  \
--opt adam \
--loss m=l2 \
--lr m=0.01 \
--mode inversion \
--save-path ./results_traditional_adam \
--use-cuda \
--grad-cut
