#export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/viscoacoustic.yml \
--gpuid 0 \
--opt adam \
--loss mse \
--mode inversion \
--use-cuda