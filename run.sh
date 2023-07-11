mpirun -n 2 \
python fwi.py config/tow.yml  \
--opt adam \
--loss ncc \
--lr vp=10.0 \
--mode inversion \
--save-path /mnt/data/wangsw/inversion/elastic_marmousi/results/tow \
--use-cuda