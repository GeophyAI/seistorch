python ../../../../seistorch/codingfwi_jax.py configs/tti_lsrtm_jax.yml  \
--opt adam \
--loss m=l2 \
--lr m=0.01 \
--batchsize 10 \
--mode inversion \
--save-path ./results/tti_bornobs_jax \
--use-cuda \
--grad-cut