export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/multiple.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path /mnt/data/wangsw/inversion/elastic_marmousi/results/fixed/multiple/l2_linear \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
