torchrun --master-port 2048 --nproc_per_node=1 ../../../../seistorch/seistorch_dist.py configs/lsrtm_born.yml  \
--opt cg \
--loss m=l2 \
--lr m=0.01 \
--mode inversion \
--save-path ./results_traditional_cg_born \
--use-cuda \
--grad-cut
