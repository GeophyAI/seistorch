python /mnt/desktop_wsw/seistorch/codingfwi.py lsrtm.yml \
--gpuid 0 \
--opt adam \
--loss m=l2 \
--mode inversion \
--batchsize 20 \
--lr m=0.1 \
--save-path ./results \
--checkpoint ./none \
--use-cuda