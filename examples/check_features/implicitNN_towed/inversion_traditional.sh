torchrun --nnodes=1 --nproc_per_node=4 \
/home/shaowinw/Desktop/seistorch/seistorch_dist.py \
config/forward.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed \
--use-cuda