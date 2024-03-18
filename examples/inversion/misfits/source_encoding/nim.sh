export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../../codingfwi.py forward.yml \
--gpuid 0 \
--opt adam \
--loss vp=nim \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path ./nim \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
