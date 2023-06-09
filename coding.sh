# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding.yml \
--gpuid 0 \
--opt adam \
--loss envelope \
--mode inversion \
--batchsize 20 \
--global-lr 5 \
--save-path /mnt/data/wangsw/inversion/marmousi/acoustic/envelope/ \
--use-cuda \
--grad-cut
# --grad-smooth
