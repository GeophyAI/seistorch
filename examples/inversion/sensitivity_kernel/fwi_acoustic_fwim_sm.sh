torchrun --nproc_per_node=1 \
../../../seistorch_dist.py configs/inversion_acoustic_fwim_habc_sm.yml \
--opt adam \
--loss vp=l2 rx=l2 rz=l2 \
--lr vp=0.00 rx=0.00 rz=0.00 \
--mode inversion \
--save-path ./kernels/acoustic_fwim_habc_sm \
--use-cuda