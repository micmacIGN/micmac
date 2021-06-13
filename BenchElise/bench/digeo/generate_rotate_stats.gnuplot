set terminal jpeg
set output "image_size_times.jpg"
set key left top
plot "output/stats.txt" using 1:2 title "min", "output/stats.txt" using 1:3 title "max", "output/stats.txt" using 1:4 title "mean"
