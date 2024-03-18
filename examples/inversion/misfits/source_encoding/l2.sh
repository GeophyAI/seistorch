export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../../codingfwi.py forward.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 1 \
--lr vp=10.0 \
--save-path ./l2_1shot \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
