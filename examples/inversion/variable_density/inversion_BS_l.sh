torchrun --nproc_per_node=1 \
../../../seistorch_dist.py configs/inversion_rho.yml \
--opt adam \
--loss vp=l2 rho=l2 \
--lr vp=10.0 rho=2.0 \
--mode inversion \
--save-path ./BS \
--use-cuda