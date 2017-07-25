all:
	make -C "Algorithms/BurrowsWheelerTransform/"
	make -C "Algorithms/MoveToFront/"
	make -C "Algorithms/MultiTableHuffman/CUDA/"
	make -C "Algorithms/MultiTableHuffman/Serial/"
	make -C "Algorithms/RunLengthEncoding/"
bwt:
	make -C "Algorithms/BurrowsWheelerTransform/"
mtf:
	make -C "Algorithms/MoveToFront/"
cuda:
	make -C "Algorithms/MultiTableHuffman/CUDA/"
huffman:
	make -C "Algorithms/MultiTableHuffman/Serial/"
rle:
	make -C "Algorithms/RunLengthEncoding/"
clean:
	rm Bin/*
