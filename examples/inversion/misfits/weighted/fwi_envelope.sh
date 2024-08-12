torchrun --nproc_per_node=1 ../../../seistorch_dist.py configs/inversion_ADPML.yml  \
--opt adam \
--loss vp=envelope \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/envelope \
--use-cuda