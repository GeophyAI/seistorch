torchrun --nproc_per_node=1 ../../../../seistorch/seistorch_dist.py lsrtm.yml  \
--opt cg \
--loss m=l2 \
--lr m=0.01 \
--mode inversion \
--save-path ./results_traditional_cg \
--use-cuda \
--grad-cut
