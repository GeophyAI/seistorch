torchrun --nproc_per_node=4 --master-port 2345 ../../../../seistorch_dist.py config/inversion_l2.yml \
--opt steepestdescent \
--loss vp=l2 \
--num-batches 1 \
--lr vp=30.0 \
--mode inversion \
--save-path ./results/l2 \
--grad-cut \
--use-cuda