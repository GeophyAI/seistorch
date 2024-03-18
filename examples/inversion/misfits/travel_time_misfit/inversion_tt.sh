torchrun --nproc_per_node=4 ../../../../seistorch_dist.py config/inversion_tt.yml \
--opt steepestdescent \
--loss vp=traveltime \
--num-batches 1 \
--lr vp=30.0 \
--mode inversion \
--save-path ./results/traveltime \
--grad-cut \
--use-cuda