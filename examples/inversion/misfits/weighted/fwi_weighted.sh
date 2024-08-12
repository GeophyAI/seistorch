torchrun --nproc_per_node=1 ../../../seistorch_dist.py configs/inversion_ADPML.yml  \
--opt adam \
--loss l2=1 envelope=0.1 \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/weighted \
--use-cuda