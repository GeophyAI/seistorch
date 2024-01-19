torchrun --nproc_per_node=4 \
/home/shaowinw/Desktop/seistorch/seistorch_dist.py \
config/inversion_nn.yml \
--opt adam \
--loss vp=l2 \
--lr vp=0.0001 \
--mode inversion \
--save-path ./results/towed \
--use-cuda