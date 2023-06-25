# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_ot.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 vs=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 vs=5.78 \
--save-path /mnt/data/wangsw/inversion/overthrust_15m/compare_loss/l2 \
--use-cuda \
--grad-cut
# --grad-smooth

