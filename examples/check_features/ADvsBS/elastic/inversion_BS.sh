torchrun --nproc_per_node=1 /home/shaowinw/seistorch/seistorch_dist.py configs/inversion_BS.yml  \
--opt adam \
--loss vp=l2 vs=l2 rho=l2 \
--num-batches 1 \
--lr vp=10.0 vs=5.0 rho=3.0 \
--mode inversion \
--save-path ./results/fwi_classic_BS \
--use-cuda