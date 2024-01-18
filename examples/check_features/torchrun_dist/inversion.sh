torchrun --nproc_per_node=4 \
/home/shaowinw/Desktop/seistorch/seistorch_dist.py \
configs/forward.yml \
--opt adam \
--loss vp=l2 \
--num-batches 4 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed \
--grad-cut \
--use-cuda