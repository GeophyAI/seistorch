# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_elastic.yml \
--gpuid 0 \
--opt adam \
--loss test \
--mode inversion \
--batchsize 30 \
--global-lr 5 \
--save-path /mnt/data/wangsw/inversion/bp/compare_loss/wd \
--use-cuda \
--grad-cut
# --grad-smooth
