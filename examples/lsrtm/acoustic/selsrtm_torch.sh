python ../../../codingfwi.py configs/lsrtm_born.yml \
--gpuid 0 \
--opt adam \
--loss m=l2 \
--mode inversion \
--batchsize 10 \
--lr m=0.01 \
--save-path ./results/se_torch \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
