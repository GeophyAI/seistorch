torchrun --nproc_per_node=1 ../../../../seistorch/seistorch_dist.py configs/vti_lsrtm.yml  \
--opt adam \
--loss m=l2 \
--lr m=0.01 \
--mode inversion \
--save-path ./results/vti_adam_bornobs \
--use-cuda \
--grad-cut