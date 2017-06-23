/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900009
#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../Headers/huffman.h"
#include "../Headers/ubwt.h"
#include "../Headers/umtf.h"

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpuTimeUsed;

	//file information
	unsigned int compressedBlockLenth, inputBlockLength;
	unsigned int frequency[256];
	unsigned char inputBlockData[2 * BLOCK_SIZE];
	unsigned char huffmanOutputData[BLOCK_SIZE];
  unsigned char mtfOutputData[BLOCK_SIZE];
	unsigned char bwtOutputData[BLOCK_SIZE - 9];
	unsigned int outputFileLength;

	//structure to hold dictionary data
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];

	//mpi data
	unsigned int rank, numProcesses;
	unsigned int numCompressedBlocks;
	unsigned int mpiCompressedBlockLength, mpiOutputBlockLength;
	unsigned int *mpiCompressedBlockIndex;
	unsigned char *mpiOutputData, *mpiCompressedData;
	
	MPI_Init( &argc, &argv);
	MPI_File mpiCompressedFile, mpiOutputFile;
	MPI_Status status;
	
	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	// find length of compressedFile file
	mpiCompressedBlockIndex = (unsigned int *)malloc((numProcesses + 1) * sizeof(unsigned int));

	
	if(rank == 0){
		FILE *compressedFile;
		unsigned int compressedFileLength;
		compressedFile = fopen(argv[1], "rb");
		fseek(compressedFile, 0, SEEK_END);
		compressedFileLength = ftell(compressedFile);
		fseek(compressedFile, 0, SEEK_SET);
		fread(&outputFileLength, sizeof(unsigned int), 1, compressedFile);
		fread(&numCompressedBlocks, sizeof(unsigned int), 1, compressedFile);
		fread(mpiCompressedBlockIndex, sizeof(unsigned int), numCompressedBlocks, compressedFile);
		mpiCompressedBlockIndex[numCompressedBlocks] = compressedFileLength;
		fclose(compressedFile);
	}
	
	// broadcast block sizes, outputFileLength no. of blocks
	MPI_Bcast(&outputFileLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(mpiCompressedBlockIndex, (numProcesses + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// start clock
	if(rank == 0){
		start = clock();
	}
	
	// calculate size of block
	mpiCompressedBlockLength = mpiCompressedBlockIndex[rank + 1] - mpiCompressedBlockIndex[rank];
	
	//get output block length
	mpiOutputBlockLength = outputFileLength / numProcesses;
	if(rank == (numProcesses - 1)){
		mpiOutputBlockLength = outputFileLength - ((numProcesses - 1) * mpiOutputBlockLength);
	}

	//allocate data
	mpiCompressedData = (unsigned char *)malloc((mpiCompressedBlockLength) * sizeof(unsigned char));
	mpiOutputData = (unsigned char *)malloc(mpiOutputBlockLength * sizeof(unsigned char));

	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &mpiCompressedFile);
	MPI_File_seek(mpiCompressedFile, mpiCompressedBlockIndex[rank], MPI_SEEK_SET);
	MPI_File_read(mpiCompressedFile, mpiCompressedData, mpiCompressedBlockLength, MPI_UNSIGNED_CHAR, &status);
	
	//perform reverse BZIP2
	//convert this loop
	unsigned int blockLength = mpiCompressedBlockLength;
	unsigned char *inputBlockPointer = mpiCompressedData;
	mpiOutputBlockLength = 0;
	while( blockLength > 0 ){
		memcpy(&compressedBlockLenth, inputBlockPointer, 4);
		memcpy(&inputBlockLength, (inputBlockPointer + 4), 4);
		memcpy(frequency, (inputBlockPointer + 8), 1024);
		memcpy(inputBlockData, (inputBlockPointer + 1032), compressedBlockLenth);
	
		//perform huffman
		reverse_huffman_encoding(frequency, compressedBlockLenth, inputBlockData, huffmanOutputData);
	
		//perform reverse MTF
		reverse_move_to_front(inputBlockLength, &head, &tail, dictionaryLinkedList, huffmanOutputData, mtfOutputData);
			
		//perform reverse BWT
    reverse_burrows_wheeler_transform(inputBlockLength, mtfOutputData, bwtOutputData);
		
		//write to output
		memcpy(&mpiOutputData[mpiOutputBlockLength], bwtOutputData, inputBlockLength - 9);
		mpiOutputBlockLength += (inputBlockLength - 9);
		inputBlockPointer += (1032 + compressedBlockLenth);
		blockLength -= (1032 + compressedBlockLenth);
	}

	// get time
	if(rank == 0){
		end = clock();
		cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
	}
	
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpiOutputFile);
	MPI_File_seek(mpiOutputFile, rank * (outputFileLength / numProcesses), MPI_SEEK_SET);
	MPI_File_write(mpiOutputFile, mpiOutputData, mpiOutputBlockLength, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	//close open files
	MPI_File_close(&mpiCompressedFile); 	
	MPI_File_close(&mpiOutputFile);
	
	free(mpiCompressedBlockIndex);
	free(mpiOutputData);
	free(mpiCompressedData);	
	MPI_Finalize();
	return 0;
}
