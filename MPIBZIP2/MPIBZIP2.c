/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900000
#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../Headers/huffman.h"
#include "../Headers/bwt.h"
#include "../Headers/mtf.h"

int main(int argc, char **argv){
  //time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;

  //structure to store each symbols
	linked_list *head = NULL, *tail = NULL, dictionaryLinkedList[256];
  unsigned char bwtOutputData[BLOCK_SIZE + 9];
  unsigned char mtfOutputData[BLOCK_SIZE + 9];
  unsigned char huffmanOutputData[BLOCK_SIZE * 2];
  unsigned int frequency[256];

	//for mpi
	unsigned int rank, numProcesses;
	unsigned int mpiInputBlockLength, mpiCompressedBlockLength;
	unsigned int *mpiCompressedBlockIndex;
	unsigned char *mpiInputData, *mpiCompressedData;
  //input file length that will be bcasted from root process
	unsigned int inputFileLength;

	//initialize mpi
	MPI_Init( &argc, &argv);
	MPI_File mpiInputFile, mpiCompressedFile;
	MPI_Status status;

	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	// get file size
	if(rank == 0){
		FILE *inputFile;
		inputFile = fopen(argv[1], "rb");
		fseek(inputFile, 0, SEEK_END);
		inputFileLength = ftell(inputFile);
		fseek(inputFile, 0, SEEK_SET);
		fclose(inputFile);
	}

	//broadcast size of file to all the processes 
	MPI_Bcast(&inputFileLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get file chunk size
	mpiInputBlockLength = inputFileLength / numProcesses;
	if(rank == (numProcesses - 1)){
		mpiInputBlockLength = inputFileLength - ((numProcesses - 1) * mpiInputBlockLength);	
	}

	//allocate memory 
	mpiInputData = (unsigned char *)malloc(mpiInputBlockLength * sizeof(unsigned char));
	mpiCompressedData = (unsigned char *)malloc((mpiInputBlockLength * 2 + 1024) * sizeof(unsigned char));
	mpiCompressedBlockIndex = (unsigned int *)malloc(numProcesses * sizeof(unsigned int));

	// open file in each process and read data
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &mpiInputFile);
	MPI_File_seek(mpiInputFile, rank * (inputFileLength / numProcesses), MPI_SEEK_SET);
	MPI_File_read(mpiInputFile, mpiInputData, mpiInputBlockLength, MPI_UNSIGNED_CHAR, &status);

	// start clock
	if(rank == 0){
		start = clock();
	}

	//perform BZIP2
	mpiCompressedBlockLength = 0;
	unsigned char *inputBlockPointer = mpiInputData;
	unsigned int processLength = mpiInputBlockLength;
  while(processLength != 0){
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

	  //write to output
	  memcpy(&mpiCompressedData[mpiCompressedBlockLength], &compressedBlockLength, 4);
	  memcpy(&mpiCompressedData[mpiCompressedBlockLength + 4], &newInputBlockLength, 4);
	  memcpy(&mpiCompressedData[mpiCompressedBlockLength + 8], frequency, 1024);
	  memcpy(&mpiCompressedData[mpiCompressedBlockLength + 1032], huffmanOutputData, compressedBlockLength);
	  mpiCompressedBlockLength += compressedBlockLength + 1032;
  }
	// calculate length of compressed data
	mpiCompressedBlockIndex[rank] = mpiCompressedBlockLength;

	// send the length of each process to process 0
	MPI_Gather(&mpiCompressedBlockLength, 1, MPI_UNSIGNED, mpiCompressedBlockIndex, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// update the data to reflect the offset
	if(rank == 0){
		mpiCompressedBlockIndex[0] = (numProcesses + 2) * 4 + mpiCompressedBlockIndex[0];
		for(unsigned int i = 1; i < numProcesses; i++){
			mpiCompressedBlockIndex[i] = mpiCompressedBlockIndex[i] + mpiCompressedBlockIndex[i - 1];
		}
		for(unsigned int i = (numProcesses - 1); i > 0; i--){
			mpiCompressedBlockIndex[i] = mpiCompressedBlockIndex[i - 1];
		}
		mpiCompressedBlockIndex[0] = (numProcesses + 2) * 4;
	}

	// broadcast size of each compressed data block to all the processes 
	MPI_Bcast(mpiCompressedBlockIndex, numProcesses, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get time
	if(rank == 0){
		end = clock();
		cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
	}
	
	// write data to file
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpiCompressedFile);

	if(rank == 0){
		MPI_File_write(mpiCompressedFile, &inputFileLength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpiCompressedFile, &numProcesses, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpiCompressedFile, mpiCompressedBlockIndex, numProcesses, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(mpiCompressedFile, mpiCompressedBlockIndex[rank], MPI_SEEK_SET);
	MPI_File_write(mpiCompressedFile, mpiCompressedData, mpiCompressedBlockLength, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

	// close open files
	MPI_File_close(&mpiCompressedFile);
	MPI_File_close(&mpiInputFile);
	MPI_Barrier(MPI_COMM_WORLD);
	
	free(mpiCompressedBlockIndex);
	free(mpiInputData);
	free(mpiCompressedData);
	MPI_Finalize();
	return 0;
}



