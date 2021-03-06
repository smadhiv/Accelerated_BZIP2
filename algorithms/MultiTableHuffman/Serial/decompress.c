/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//huffman serial Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include "../../../library/huffman/serial/uhuffman_serial.h"

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int inputBlockLength, outputBlockLength, frequency[256];
	unsigned char inputBlockData[BLOCK_SIZE * 2], outputBlockData[BLOCK_SIZE];
	FILE *inputFile, *outputFile;
	
 //check for required arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }
	
	// open source compressed file
	inputFile = fopen(argv[1], "rb");
	outputFile = fopen(argv[2], "wb");
	
	// start time measure
	start = clock();
	
	//process each block
	int ret;
	while((ret = fread(&inputBlockLength, sizeof(unsigned int), 1, inputFile))){
		ret = fread(&outputBlockLength, sizeof(unsigned int), 1, inputFile);
	  ret = fread(frequency, 256 * sizeof(unsigned int), 1, inputFile);
	  ret = fread(inputBlockData, sizeof(unsigned char), (inputBlockLength), inputFile);

		//do reverse huffman
		reverse_huffman_encoding(frequency, inputBlockLength, inputBlockData, outputBlockData);
		fwrite(outputBlockData, sizeof(unsigned char), outputBlockLength, outputFile);
	}

	//display runtime
	end = clock();

	//close files
	fclose(inputFile);
	fclose(outputFile);

	//compute and print run time
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	return 0;
}
