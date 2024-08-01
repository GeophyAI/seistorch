python ../../../../codingfwi.py gaussian.yml \
--gpuid 0 \
--opt adam \
--loss vp=l1 \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 \
--save-path ./results/l1_gaussian \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
