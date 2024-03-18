export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../../codingfwi.py inversion_random.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 1 \
--lr vp=10.0 \
--save-path ./results_rand \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
