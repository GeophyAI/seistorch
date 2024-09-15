python ../../../../seisjax.py configs/inversion_ADPML_jax.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 1 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/fwi_classic_ADPML_jax \
--use-cuda