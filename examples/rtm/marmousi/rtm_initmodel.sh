torchrun --nproc_per_node=1 ../../../seistorch_dist.py rtm_initmodel.yml  \
--opt adam \
--loss vp=rtm \
--num-batches 87 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/rtm_init \
--use-cuda