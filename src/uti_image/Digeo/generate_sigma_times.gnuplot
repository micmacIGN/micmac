set terminal jpeg
set output "sigma_times.jpg"
set key left top
plot "sigma_convolution_times.txt" using 1:5 title "compiled", "sigma_convolution_times.txt" using 1:6 title "not compiled", "sigma_convolution_times.txt" using 1:7 title "legacy"
