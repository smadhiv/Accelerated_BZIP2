/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//huffman serial Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../../Headers/huffman_serial.h"
#define BLOCK_SIZE 900000

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
	while(fread(&inputBlockLength, sizeof(unsigned int), 1, inputFile)){
		fread(&outputBlockLength, sizeof(unsigned int), 1, inputFile);
	  fread(frequency, 256 * sizeof(unsigned int), 1, inputFile);
	  fread(inputBlockData, sizeof(unsigned char), (inputBlockLength), inputFile);

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
