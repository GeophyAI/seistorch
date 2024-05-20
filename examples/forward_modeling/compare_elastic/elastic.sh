mpirun -f hosts \
python ../../../fwi.py configs/elastic.yml \
--mode forward \
--num-batches 5 \
--use-cuda