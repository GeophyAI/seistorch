python ../../../../codingfwi_jax.py configs/inversion_jax.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 \
--save-path ./results/jax \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
