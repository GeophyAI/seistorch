# export LD_LIBRARY_PATH=/home/les_01/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_elastic.yml \
--gpuid 0 \
--opt adam \
--loss vp=envelope vs=envelope \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 vs=5.780346820809249 \
--save-path /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/envelope_nonorm \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth

