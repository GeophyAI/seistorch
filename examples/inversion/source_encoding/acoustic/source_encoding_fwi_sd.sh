python /home/shaowinw/seistorch/codingfwi.py forward.yml \
--gpuid 0 \
--opt steepestdescent \
--loss vp=cs \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path ./results_sd \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
