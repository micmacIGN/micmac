set terminal jpeg
set output "image_size_times.jpg"
set key left top
plot "image_size_convolution_times.txt.U_INT1.txt" using 2:5 title "compiled uint1", "image_size_convolution_times.txt.U_INT1.txt" using 2:6 title "not compiled uint1", "image_size_convolution_times.txt.U_INT1.txt" using 2:7 title "legacy uint1"
