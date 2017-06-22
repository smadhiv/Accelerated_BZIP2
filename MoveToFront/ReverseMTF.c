/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//revere MTF Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900000
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Headers/umtf.h"

int main(int argc, char **argv){
	//time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;
	//file information
	unsigned int inputBlockLength;
	unsigned char inputFileData[BLOCK_SIZE], outputDataIndex[BLOCK_SIZE];
	FILE *inputFile, *outFile;
	//structure to hold dictionary data
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];

	// check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	// start time measure
	start = clock();

	//read input file, output file
	inputFile = fopen(argv[1], "rb");
	outFile = fopen(argv[2], "wb");

	while( (inputBlockLength = fread(inputFileData, sizeof(unsigned char), BLOCK_SIZE, inputFile)) ){
    reverse_move_to_front(inputBlockLength, &head, &tail, dictionaryLinkedList, inputFileData, outputDataIndex);
		//write output  data
		fwrite(outputDataIndex, sizeof(unsigned char), inputBlockLength, outFile);
	}

	//close files
	fclose(inputFile);
	fclose(outFile);
	
	// end time measure
	end = clock();
	cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
  return 0;
}
