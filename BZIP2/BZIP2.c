/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//bzip2 Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../library/huffman/serial/huffman_serial.h"
#include "../library/bwt/bwt.h"
#include "../library/mtf/mtf.h"

//global variables to be used in qsort function
extern unsigned int inputBlockLength;
//store input block data
extern unsigned char inputBlockData[BLOCK_SIZE];

int main(int argc, char **argv){
  //time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;
  //files for i/o
  FILE *inputFile, *outputFile;
  //structure to store each symbols
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];
  unsigned char bwtOutputData[BLOCK_SIZE + 9];
  unsigned char mtfOutputData[BLOCK_SIZE + 9];
  unsigned char huffmanOutputData[BLOCK_SIZE * 2];
  //compressed data
  unsigned int frequency[256];
  
  //check for required arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open i/o file
  inputFile = fopen(argv[1], "rb");
  outputFile = fopen(argv[2], "wb");

	// start time measure
	start = clock();

  //perform BWT in the while loop
  //read BLOCK_SIZE at a time, stop when there is nothing to read
  //store output data
  while( (inputBlockLength = fread(inputBlockData, 1, BLOCK_SIZE, inputFile)) ){
    //perform BWT
    burrows_wheeler_transform(bwtOutputData);
    unsigned int newInputBlockLength = inputBlockLength + 9;
    //perform MTF 
    move_to_front(newInputBlockLength, &head, &tail, dictionaryLinkedList, bwtOutputData, mtfOutputData);
    //perform huffman
    unsigned int compressedBlockLength = huffman_encoding(frequency, newInputBlockLength, mtfOutputData, huffmanOutputData);

		//write to output
    fwrite(&compressedBlockLength, sizeof(unsigned int), 1, outputFile);
    fwrite(&newInputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency, sizeof(unsigned int), 256, outputFile);
		fwrite(huffmanOutputData, sizeof(unsigned char), compressedBlockLength, outputFile);
  }

	// end time measure
	end = clock();
  
  //close i/o files
  fclose(inputFile);
  fclose(outputFile);

  //compute and print run time
  cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
  return 0;
}

