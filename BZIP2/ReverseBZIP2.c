/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//reverse bzip2
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900009
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "header/huffman.h"
#include "header/ubwt.h"
#include "header/umtf.h"

int main(int argc, char **argv){
	//time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;

	//file information
	unsigned int compressedBlockLenth, inputBlockLength;
	unsigned int frequency[256];
	unsigned char inputBlockData[2 * BLOCK_SIZE];
	unsigned char huffmanOutputData[BLOCK_SIZE];
  unsigned char mtfOutputData[BLOCK_SIZE];
	unsigned char bwtOutputData[BLOCK_SIZE - 9];
	
	FILE *inputFile, *outFile;
	//structure to hold dictionary data
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];

	// check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	//read input file, output file
	inputFile = fopen(argv[1], "rb");
	outFile = fopen(argv[2], "wb");

	// start time measure
	start = clock();

	while( (fread(&compressedBlockLenth, sizeof(unsigned int), 1, inputFile)) ){
		fread(&inputBlockLength, sizeof(unsigned int), 1, inputFile);
		fread(frequency, sizeof(unsigned int), 256, inputFile);
    fread(inputBlockData, sizeof(unsigned char), inputBlockLength, inputFile);
		reverse_huffman_encoding(frequency, inputBlockLength, inputBlockData, huffmanOutputData);
		//perform reverse MTF
		reverse_move_to_front(inputBlockLength, &head, &tail, dictionaryLinkedList, huffmanOutputData, mtfOutputData);
		//perform reverse BWT
    reverse_burrows_wheeler_transform(inputBlockLength, mtfOutputData, bwtOutputData);
		fwrite(bwtOutputData, sizeof(unsigned char), inputBlockLength - 9, outFile);
	}
	
	// end time measure
	end = clock();

	//close files
	fclose(inputFile);
	fclose(outFile);

	//compute and print run time
	cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
  return 0;
}
