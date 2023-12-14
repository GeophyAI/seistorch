export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../codingfwi.py config/siren.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 10 \
--lr vp=0.0001 \
--save-path ./siren_nolow2 \
--checkpoint ./none \
--use-cuda
