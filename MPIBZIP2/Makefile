all:
	mpicc -Wall -Wextra -O3 MPIBZIP2.c ../library/mtf/mtf.c ../library/bwt/bwt.c ../library/huffman/serial/huffman_serial.c -o MPIBZIP2
	mpicc -Wall -Wextra -O3 ReverseMPIBZIP2.c ../library/mtf/umtf.c ../library/bwt/ubwt.c ../library/huffman/serial/uhuffman_serial.c -o ReverseMPIBZIP2

clean:
	rm MPIBZIP2 ReverseMPIBZIP2
