mpirun -n 2 \
python fwi.py config/fixed_elastic.yml  \
--opt adam \
--loss ncc \
--lr vp=10.0 \
--mode inversion \
--save-path /mnt/data/wangsw/inversion/elastic_marmousi/results/fixed/aec/test \
--use-cuda