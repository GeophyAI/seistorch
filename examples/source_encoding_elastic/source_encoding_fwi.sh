export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../codingfwi.py forward.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 vs=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 vs=5.78 \
--save-path ./results \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
