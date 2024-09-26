python ../../../../seisjax.py inversion_jax.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--step-per-epoch 12 \
--save-path ./results/towed_jax \
--grad-cut \
--use-cuda