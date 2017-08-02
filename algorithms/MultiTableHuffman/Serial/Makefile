all: 
	gcc -std=c11 -O3 -Wall -Wextra compress.c ../../../library/huffman/serial/huffman_serial.c -o ../../../bin/serialCompress
	gcc -std=c11 -O3 -Wall -Wextra decompress.c ../../../library/huffman/serial/uhuffman_serial.c -o ../../../bin/serialDecompress

clean:
	rm ../../../bin/serialCompress ../../../bin/serialDecompress