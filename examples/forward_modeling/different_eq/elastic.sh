mpirun -f hosts \
python ../../../fwi.py configs/elastic.yml \
--mode forward \
--num-batches 1 \
--use-cuda