python /home/shaowinw/seistorch/codingfwi.py forward.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path ./results \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
