/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define MIN_SCRATCH_SIZE 50 * 1024 * 1024
#define BLOCK_SIZE 900000
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../Headers/huffman.h"
#include "../Headers/bwt.h"
#include "../Headers/mtf.h"
#include "../Headers/parallelHeader.h"

unsigned char bitSequenceConstMemory[256][255];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;

int main(int argc, char **argv){
	//time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;
	//files for i/o
	FILE *inputFile, *compressedFile, *outputFile;
  //structure to store each symbols
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];
	unsigned char bwtOutputData[BLOCK_SIZE + 9];
	unsigned char mtfOutputData[BLOCK_SIZE + 9];
	unsigned char huffmanOutputData[BLOCK_SIZE * 2];
	unsigned int frequency[256];
	//cuda huffman specific data
	unsigned int i;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, frequency[256];
	unsigned char *inputFileData, bitSequenceLength = 0, bitSequence[255];
	unsigned int *compressedDataOffset, cpu_time_used;
	unsigned int integerOverflowFlag;

	long unsigned int mem_free, mem_total;
	long unsigned int mem_req, mem_offset, mem_data;
	int numKernelRuns;
	
  //check for required arguments
	if(argc != 3){
		printf("Error: Need input and output file names\n");
		return -1;
	}

	// start time measure
	start = clock();

	// read input file, get inputFileLength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);

  //perform BWT in the while loop
  //read BLOCK_SIZE at a time, stop when there is nothing to read
  //store output data
	unsigned int compressedBlockLength = 0;
	unsigned char *inputBlockPointer = inputFileData;
	unsigned int processLength = inputFileLength;
  while( processLength != 0 ){
		inputBlockLength = processLength > BLOCK_SIZE ? BLOCK_SIZE : processLength;
	  processLength -= inputBlockLength;

		//copy input data to global memory
	  memcpy(inputBlockData, inputBlockPointer, inputBlockLength);
		inputBlockPointer += inputBlockLength;
    //perform BWT
    burrows_wheeler_transform(bwtOutputData);
    unsigned int newInputBlockLength = inputBlockLength + 9;
    //perform MTF 
    move_to_front(newInputBlockLength, &head, &tail, dictionaryLinkedList, bwtOutputData, mtfOutputData);
    //perform huffman
    unsigned int compressedBlockLength = huffman_encoding(frequency, newInputBlockLength, mtfOutputData, huffmanOutputData);

//***************************************************************************/
//process from next line
		//write to output
    fwrite(&compressedBlockLength, sizeof(unsigned int), 1, outputFile);
    fwrite(&newInputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency, sizeof(unsigned int), 256, outputFile);
		fwrite(huffmanOutputData, sizeof(unsigned char), compressedBlockLength, outputFile);
  }
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < inputFileLength; i++){
		frequency[inputFileData[i]]++;
	}

	// initialize nodes of huffman tree
	distinctCharacterCount = 0;
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacterCount].count = frequency[i];
			huffmanTreeNode[distinctCharacterCount].letter = i;
			huffmanTreeNode[distinctCharacterCount].left = NULL;
			huffmanTreeNode[distinctCharacterCount].right = NULL;
			distinctCharacterCount++;
		}
	}
	
	// build tree 
	for (i = 0; i < distinctCharacterCount - 1; i++){
		combinedHuffmanNodes = 2 * i;
		sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
		buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
	}
	
	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);
	
	// calculate memory requirements
	// GPU memory
	cudaMemGetInfo(&mem_free, &mem_total);
	
	// debug
	if(1){
		printf("Free Mem: %lu\n", mem_free);		
	}

	// offset array requirements
	mem_offset = 0;
	for(i = 0; i < 256; i++){
		mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
	}
	mem_offset = mem_offset % 8 == 0 ? mem_offset : mem_offset + 8 - mem_offset % 8;
	
	// other memory requirements
	mem_data = inputFileLength + (inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);
	
	if(mem_free - mem_data < MIN_SCRATCH_SIZE){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
		return -1;
	}
	mem_req = mem_free - mem_data - 10 * 1024 * 1024;
	numKernelRuns = ceil((double)mem_offset / mem_req);
	integerOverflowFlag = mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX ? 0 : 1;

	// debug
	if(1){
	printf("	InputFileSize      =%u\n\
	OutputSize         =%u\n\
	NumberOfKernel     =%d\n\
	integerOverflowFlag=%d\n", inputFileLength, mem_offset/8, numKernelRuns, integerOverflowFlag);		
	}

	
	// generate data offset array
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));

	// launch kernel
	lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, inputFileLength, numKernelRuns, integerOverflowFlag, mem_req);

	// calculate run duration
	end = clock();
	
	// write src inputFileLength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
	fwrite(inputFileData, sizeof(unsigned char), mem_offset / 8, compressedFile);
	fclose(compressedFile);	
	
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedDataOffset);
	return 0;
}
