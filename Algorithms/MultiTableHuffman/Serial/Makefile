all: 
	gcc -std=c11 -O3 -Wall -Wextra compress.c ../../../library/huffman/serial/huffman_serial.c -o ../../../Bin/serialCompress
	gcc -std=c11 -O3 -Wall -Wextra decompress.c ../../../library/huffman/serial/uhuffman_serial.c -o ../../../Bin/serialDecompress

clean:
	rm ../../../Bin/serialCompress ../../../Bin/serialDecompress