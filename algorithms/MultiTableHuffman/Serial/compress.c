/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//huffman serial Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include "../../../library/huffman/serial/huffman_serial.h"

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int frequency[256], compressedBlockLength, inputBlockLength;
	unsigned char inputBlockData[BLOCK_SIZE], compressedBlockData[2 * BLOCK_SIZE];
	FILE *inputFile, *compressedFile;
	
	//check for required arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

	// read input file, get filelength and data
	inputFile = fopen(argv[1], "rb");
	compressedFile = fopen(argv[2], "wb");

	// start time measure
	start = clock();
		
	while( (inputBlockLength = fread(inputBlockData, 1, BLOCK_SIZE, inputFile)) ){
		compressedBlockLength = huffman_encoding(frequency, inputBlockLength, inputBlockData, compressedBlockData);
		fwrite(&compressedBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
		fwrite(compressedBlockData, sizeof(unsigned char), compressedBlockLength, compressedFile);
	}

	// calculate run duration
	end = clock();

//close files
	fclose(inputFile);
	fclose(compressedFile);

	//compute and print run time
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	return 0;
}



