#set terminal pdf color enhanced \
#dashed dl 1.0 size 20.0cm, 20.0cm 
#set output "lattice.pdf"
set xrange [-2.000000: 4.800000]
set yrange [-2.000000: 4.800000]
set size square
unset key
unset tics
unset border
set style line 1 lc 1 lt 1
set style line 2 lc 5 lt 1
set style line 3 lc 0 lt 1
set arrow from 0.000000, 0.000000 to 2.000000, 0.200000 nohead front ls 3
set arrow from 2.000000, 0.200000 to 2.800000, 2.000000 nohead front ls 3
set arrow from 2.800000, 2.000000 to 0.800000, 1.800000 nohead front ls 3
set arrow from 0.800000, 1.800000 to 0.000000, 0.000000 nohead front ls 3
set label "0" at 0.000000, 0.000000 center front
set label "1" at -1.000000, -0.100000 center front
set arrow from 0.000000, 0.000000 to -1.000000, -0.100000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "1" at 1.000000, 0.100000 center front
set arrow from 0.000000, 0.000000 to 1.000000, 0.100000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "2" at -0.400000, -0.900000 center front
set arrow from 0.000000, 0.000000 to -0.400000, -0.900000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "2" at 0.400000, 0.900000 center front
set arrow from 0.000000, 0.000000 to 0.400000, 0.900000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "3" at -0.600000, 0.800000 center front
set arrow from 0.000000, 0.000000 to -0.600000, 0.800000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "3" at 0.600000, -0.800000 center front
set arrow from 0.000000, 0.000000 to 0.600000, -0.800000 nohead ls 1
set label "0" at 0.000000, 0.000000 center front
set label "2" at -1.600000, 0.700000 center front
set arrow from 0.000000, 0.000000 to -1.600000, 0.700000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "2" at 1.600000, -0.700000 center front
set arrow from 0.000000, 0.000000 to 1.600000, -0.700000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "3" at -1.400000, -1.000000 center front
set arrow from 0.000000, 0.000000 to -1.400000, -1.000000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "3" at 1.400000, 1.000000 center front
set arrow from 0.000000, 0.000000 to 1.400000, 1.000000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "1" at 0.200000, -1.700000 center front
set arrow from 0.000000, 0.000000 to 0.200000, -1.700000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "1" at -0.200000, 1.700000 center front
set arrow from 0.000000, 0.000000 to -0.200000, 1.700000 nohead ls 2
set label "0" at 0.000000, 0.000000 center front
set label "0" at -2.000000, -0.200000 center front
set label "0" at 0.000000, 0.000000 center front
set label "0" at 2.000000, 0.200000 center front
set label "0" at 0.000000, 0.000000 center front
set label "0" at -0.800000, -1.800000 center front
set label "0" at 0.000000, 0.000000 center front
set label "0" at 0.800000, 1.800000 center front
set label "0" at 0.000000, 0.000000 center front
set label "0" at -1.200000, 1.600000 center front
set label "0" at 0.000000, 0.000000 center front
set label "0" at 1.200000, -1.600000 center front
set label "1" at 1.000000, 0.100000 center front
set label "0" at 0.000000, 0.000000 center front
set arrow from 1.000000, 0.100000 to 0.000000, 0.000000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "0" at 2.000000, 0.200000 center front
set arrow from 1.000000, 0.100000 to 2.000000, 0.200000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "3" at 0.600000, -0.800000 center front
set arrow from 1.000000, 0.100000 to 0.600000, -0.800000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "3" at 1.400000, 1.000000 center front
set arrow from 1.000000, 0.100000 to 1.400000, 1.000000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "2" at 0.400000, 0.900000 center front
set arrow from 1.000000, 0.100000 to 0.400000, 0.900000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "2" at 1.600000, -0.700000 center front
set arrow from 1.000000, 0.100000 to 1.600000, -0.700000 nohead ls 1
set label "1" at 1.000000, 0.100000 center front
set label "3" at -0.600000, 0.800000 center front
set arrow from 1.000000, 0.100000 to -0.600000, 0.800000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "3" at 2.600000, -0.600000 center front
set arrow from 1.000000, 0.100000 to 2.600000, -0.600000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "2" at -0.400000, -0.900000 center front
set arrow from 1.000000, 0.100000 to -0.400000, -0.900000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "2" at 2.400000, 1.100000 center front
set arrow from 1.000000, 0.100000 to 2.400000, 1.100000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "0" at 1.200000, -1.600000 center front
set arrow from 1.000000, 0.100000 to 1.200000, -1.600000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "0" at 0.800000, 1.800000 center front
set arrow from 1.000000, 0.100000 to 0.800000, 1.800000 nohead ls 2
set label "1" at 1.000000, 0.100000 center front
set label "1" at -1.000000, -0.100000 center front
set label "1" at 1.000000, 0.100000 center front
set label "1" at 3.000000, 0.300000 center front
set label "1" at 1.000000, 0.100000 center front
set label "1" at 0.200000, -1.700000 center front
set label "1" at 1.000000, 0.100000 center front
set label "1" at 1.800000, 1.900000 center front
set label "1" at 1.000000, 0.100000 center front
set label "1" at -0.200000, 1.700000 center front
set label "1" at 1.000000, 0.100000 center front
set label "1" at 2.200000, -1.500000 center front
set label "2" at 0.400000, 0.900000 center front
set label "3" at -0.600000, 0.800000 center front
set arrow from 0.400000, 0.900000 to -0.600000, 0.800000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "3" at 1.400000, 1.000000 center front
set arrow from 0.400000, 0.900000 to 1.400000, 1.000000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "0" at 0.000000, 0.000000 center front
set arrow from 0.400000, 0.900000 to 0.000000, 0.000000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "0" at 0.800000, 1.800000 center front
set arrow from 0.400000, 0.900000 to 0.800000, 1.800000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "1" at -0.200000, 1.700000 center front
set arrow from 0.400000, 0.900000 to -0.200000, 1.700000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "1" at 1.000000, 0.100000 center front
set arrow from 0.400000, 0.900000 to 1.000000, 0.100000 nohead ls 1
set label "2" at 0.400000, 0.900000 center front
set label "0" at -1.200000, 1.600000 center front
set arrow from 0.400000, 0.900000 to -1.200000, 1.600000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "0" at 2.000000, 0.200000 center front
set arrow from 0.400000, 0.900000 to 2.000000, 0.200000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "1" at -1.000000, -0.100000 center front
set arrow from 0.400000, 0.900000 to -1.000000, -0.100000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "1" at 1.800000, 1.900000 center front
set arrow from 0.400000, 0.900000 to 1.800000, 1.900000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "3" at 0.600000, -0.800000 center front
set arrow from 0.400000, 0.900000 to 0.600000, -0.800000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "3" at 0.200000, 2.600000 center front
set arrow from 0.400000, 0.900000 to 0.200000, 2.600000 nohead ls 2
set label "2" at 0.400000, 0.900000 center front
set label "2" at -1.600000, 0.700000 center front
set label "2" at 0.400000, 0.900000 center front
set label "2" at 2.400000, 1.100000 center front
set label "2" at 0.400000, 0.900000 center front
set label "2" at -0.400000, -0.900000 center front
set label "2" at 0.400000, 0.900000 center front
set label "2" at 1.200000, 2.700000 center front
set label "2" at 0.400000, 0.900000 center front
set label "2" at -0.800000, 2.500000 center front
set label "2" at 0.400000, 0.900000 center front
set label "2" at 1.600000, -0.700000 center front
set label "3" at 1.400000, 1.000000 center front
set label "2" at 0.400000, 0.900000 center front
set arrow from 1.400000, 1.000000 to 0.400000, 0.900000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "2" at 2.400000, 1.100000 center front
set arrow from 1.400000, 1.000000 to 2.400000, 1.100000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "1" at 1.000000, 0.100000 center front
set arrow from 1.400000, 1.000000 to 1.000000, 0.100000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "1" at 1.800000, 1.900000 center front
set arrow from 1.400000, 1.000000 to 1.800000, 1.900000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "0" at 0.800000, 1.800000 center front
set arrow from 1.400000, 1.000000 to 0.800000, 1.800000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "0" at 2.000000, 0.200000 center front
set arrow from 1.400000, 1.000000 to 2.000000, 0.200000 nohead ls 1
set label "3" at 1.400000, 1.000000 center front
set label "1" at -0.200000, 1.700000 center front
set arrow from 1.400000, 1.000000 to -0.200000, 1.700000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "1" at 3.000000, 0.300000 center front
set arrow from 1.400000, 1.000000 to 3.000000, 0.300000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "0" at 0.000000, 0.000000 center front
set arrow from 1.400000, 1.000000 to 0.000000, 0.000000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "0" at 2.800000, 2.000000 center front
set arrow from 1.400000, 1.000000 to 2.800000, 2.000000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "2" at 1.600000, -0.700000 center front
set arrow from 1.400000, 1.000000 to 1.600000, -0.700000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "2" at 1.200000, 2.700000 center front
set arrow from 1.400000, 1.000000 to 1.200000, 2.700000 nohead ls 2
set label "3" at 1.400000, 1.000000 center front
set label "3" at -0.600000, 0.800000 center front
set label "3" at 1.400000, 1.000000 center front
set label "3" at 3.400000, 1.200000 center front
set label "3" at 1.400000, 1.000000 center front
set label "3" at 0.600000, -0.800000 center front
set label "3" at 1.400000, 1.000000 center front
set label "3" at 2.200000, 2.800000 center front
set label "3" at 1.400000, 1.000000 center front
set label "3" at 0.200000, 2.600000 center front
set label "3" at 1.400000, 1.000000 center front
set label "3" at 2.600000, -0.600000 center front
plot '-' w d lc 7
0.0 0.0
end
pause -1
