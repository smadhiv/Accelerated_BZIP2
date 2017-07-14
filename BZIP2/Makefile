all:
	gcc -std=c11 -O3 -Wall -Wextra BZIP2.c ../library/mtf/mtf.c ../library/bwt/bwt.c ../library/huffman/serial/huffman_serial.c -o BZIP2
	gcc -std=c11 -O3 -Wall -Wextra ReverseBZIP2.c ../library/mtf/umtf.c ../library/bwt/ubwt.c ../library/huffman/serial/uhuffman_serial.c -o ReverseBZIP2
clean:
	rm BZIP2 ReverseBZIP2
