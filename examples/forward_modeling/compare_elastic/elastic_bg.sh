mpirun -f hosts \
python ../../../fwi.py configs/elastic_bg.yml \
--mode forward \
--num-batches 5 \
--use-cuda