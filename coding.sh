export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/tow.yml \
--gpuid 0 \
--opt adam \
--loss vp=cs \
--mode inversion \
--batchsize 100 \
--lr vp=10.0 vs=5.780346820809249 \
--save-path /mnt/data/wangsw/inversion/elastic_marmousi/results/tow/cs \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
