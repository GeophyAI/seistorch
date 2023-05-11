export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding.yml \
--gpuid 3 \
--opt adam \
--loss envelope \
--mode inversion \
--save-path /public1/home/wangsw/FWI/EFWI/Marmousi/marmousi1_20m/compare_loss/envelope \
--use-cuda