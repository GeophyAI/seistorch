export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/field/obn_field.yml \
--gpuid 0 \
--opt adam \
--loss vp=fatt \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path /public1/home/wangsw/FWI/FIELDDATA/OBN/results/fatt2 \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
