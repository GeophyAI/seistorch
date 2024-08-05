torchrun --nproc_per_node=1 \
../../../seistorch_dist.py configs/inversion_rho_BS_a.yml \
--opt adam \
--loss vp=l2 rho=l2 \
--lr vp=0.00 rho=0.00 \
--mode inversion \
--save-path ./BSa \
--use-cuda