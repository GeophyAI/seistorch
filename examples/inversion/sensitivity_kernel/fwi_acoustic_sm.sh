torchrun --nproc_per_node=1 \
../../../seistorch_dist.py configs/inversion_acoustic_habc_sm.yml \
--opt adam \
--loss vp=l2 \
--lr vp=0.00 \
--mode inversion \
--save-path ./kernels/acoustic_habc_sm \
--use-cuda