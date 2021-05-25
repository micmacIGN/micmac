set terminal jpeg
set output "sigma_times.jpg"
set key left top
plot "sigma_convolution_times.U_INT1.txt" using 1:5 title "compiled uint1", "sigma_convolution_times.U_INT1.txt" using 1:6 title "not compiled uint1", "sigma_convolution_times.U_INT1.txt" using 1:7 title "legacy uint1",\
"sigma_convolution_times.U_INT2.txt" using 1:5 title "compiled uint2", "sigma_convolution_times.U_INT2.txt" using 1:6 title "not compiled uint2", "sigma_convolution_times.U_INT2.txt" using 1:7 title "legacy uint2",\
"sigma_convolution_times.REAL4.txt" using 1:5 title "compiled real4", "sigma_convolution_times.REAL4.txt" using 1:6 title "not compiled real4", "sigma_convolution_times.REAL4.txt" using 1:7 title "legacy real4"
