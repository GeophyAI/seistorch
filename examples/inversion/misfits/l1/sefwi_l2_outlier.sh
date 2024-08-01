python ../../../../codingfwi.py outlier.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 \
--save-path ./results/l2_outlier \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
