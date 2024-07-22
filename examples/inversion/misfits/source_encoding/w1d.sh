export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../../codingfwi.py forward.yml \
--gpuid 0 \
--opt adam \
--loss vp=w1d_test \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 \
--save-path ./w1d \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
