set title "Memory and GPU Memory Usage"
set xlabel "Time"
set ylabel "Usage (%)"
set terminal png size 800,600
set output "memory_usage.png"
plot "memory_usage.log" using 1:2 with lines title "Memory Usage", \
     "memory_usage.log" using 1:3 with lines title "GPU Memory Usage"

