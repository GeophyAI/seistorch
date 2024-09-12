python ../../../seisjax.py configs/inversion_jax.yml \
--opt adam \
--loss vp=l2 vs=l2 \
--lr vp=10.0 vs=5.78 \
--mode inversion \
--save-path ./results/jax \
--use-cuda