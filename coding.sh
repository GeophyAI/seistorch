# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_bp.yml \
--gpuid 0 \
--opt adam \
--loss niml2 \
--mode inversion \
--batchsize 11 \
--global-lr 5 \
--save-path /mnt/data/wangsw/inversion/bp/compare_loss/niml2 \
--use-cuda \
--grad-cut
# --grad-smooth
