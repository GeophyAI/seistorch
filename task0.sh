export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/coding_elastic.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 vs=l2 \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 vs=5.780346820809249 \
--save-path /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss_newckpt/l2 \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth

