all:
	make -C "algorithms/BurrowsWheelerTransform/"
	make -C "algorithms/MoveToFront/"
	make -C "algorithms/MultiTableHuffman/CUDA/"
	make -C "algorithms/MultiTableHuffman/Serial/"
	make -C "algorithms/RunLengthEncoding/"
bwt:
	make -C "algorithms/BurrowsWheelerTransform/"
mtf:
	make -C "algorithms/MoveToFront/"
cuda:
	make -C "algorithms/MultiTableHuffman/CUDA/"
huffman:
	make -C "algorithms/MultiTableHuffman/Serial/"
rle:
	make -C "algorithms/RunLengthEncoding/"
emulator:
	make -C "algorithms/MultiTableHuffman/emulator/"
clean:
	rm bin/*
