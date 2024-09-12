torchrun --nproc_per_node=1 \
../../../seistorch_dist.py configs/inversion_torch.yml \
--opt adam \
--loss vp=l2 vs=l2 \
--lr vp=10.0 vs=5.78 \
--mode inversion \
--save-path ./results/torch \
--use-cuda