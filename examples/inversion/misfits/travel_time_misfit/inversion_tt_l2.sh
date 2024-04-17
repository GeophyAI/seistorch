torchrun --nproc_per_node=4 ../../../../seistorch_dist.py config/inversion_tt_l2.yml \
--opt steepestdescent \
--loss vp=l2 \
--num-batches 1 \
--lr vp=30.0 \
--mode inversion \
--save-path ./results/traveltime_l2 \
--grad-cut \
--use-cuda