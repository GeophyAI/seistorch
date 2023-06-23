# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_elastic.yml \
--gpuid 0 \
--opt adam \
--loss l2 \
--mode inversion \
--batchsize 20 \
--global-lr 10 \
--save-path /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss/l2 \
--use-cuda \
--grad-cut
# --grad-smooth

