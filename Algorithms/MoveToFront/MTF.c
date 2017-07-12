/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MTF Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900000
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../Headers/mtf.h"

int main(int argc, char **argv){
	//to measure time
	clock_t start, end;
	unsigned int cpuTimeUsed;
	//file information
	unsigned int inputBlockLength;
	unsigned char inputFileData[BLOCK_SIZE], outputDataIndex[BLOCK_SIZE];
	FILE *inputFile, *outFile;
	//structure to store each symbols
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];

	//check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	//start time measure
	start = clock();

	//open input file, output file
	inputFile = fopen(argv[1], "rb");
	outFile = fopen(argv[2], "wb");
	
	//read one block at a time, process and write to output
	while( (inputBlockLength = fread(inputFileData, sizeof(unsigned char), BLOCK_SIZE, inputFile)) ){

    //perform MTF on the block
    move_to_front(inputBlockLength, &head, &tail, dictionaryLinkedList, inputFileData, outputDataIndex);
		//write to output
		//1. output data which is an array of indeces
		fwrite(outputDataIndex, sizeof(unsigned char), inputBlockLength, outFile);
	}

	// end time measure
	end = clock();
	cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
	
	//close input file
	fclose(inputFile);	
	fclose(outFile);
  return 0;
}
