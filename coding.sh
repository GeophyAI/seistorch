# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding.yml \
--gpuid 0 \
--opt adam \
--loss l2 \
--mode inversion \
--save-path /mnt/data/wangsw/inversion/marmousi_10m/compare_init/good_init_lowfreq \
--use-cuda