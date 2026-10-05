#set terminal pdf color enhanced \
#dashed dl 1.0 size 20.0cm, 20.0cm 
#set output "lattice.pdf"
set xrange [-2.000000: 6.500000]
set yrange [-2.000000: 6.500000]
set size square
unset key
unset tics
unset border
set style line 1 lc 1 lt 1
set style line 2 lc 5 lt 1
set style line 3 lc 0 lt 1
set arrow from 0.000000, 0.000000 to 3.000000, 0.000000 nohead front ls 3
set arrow from 3.000000, 0.000000 to 4.500000, 2.598076 nohead front ls 3
set arrow from 4.500000, 2.598076 to 1.500000, 2.598076 nohead front ls 3
set arrow from 1.500000, 2.598076 to 0.000000, 0.000000 nohead front ls 3
