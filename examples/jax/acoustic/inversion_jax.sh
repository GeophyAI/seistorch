python ../../../seisjax.py configs/inversion.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/jax \
--grad-cut \
--use-cuda