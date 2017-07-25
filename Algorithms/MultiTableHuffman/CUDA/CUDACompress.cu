/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#ifndef huffman_parallel
#include "../../../library/huffman/parallel/huffman_parallel.h"
#endif

//global variables to be used in qsort function
//extern unsigned int inputBlockLength;
//store input block data
//remove this for cudabzip2
unsigned char inputBlockData[BLOCK_SIZE];
unsigned char bitSequenceConstMemory[256][255];
//unsigned int constMemoryFlag = 0;

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int **frequency, compressedBlockLength, inputBlockLength;
	unsigned int inputFileLength;
	unsigned char *inputFileData, *compressedData;

	FILE *inputFile, *compressedFile;

	//gpu specific
	unsigned int *compressedDataOffset;
	//unsigned int integerOverflowFlag;
	//long unsigned int mem_req;
	//int numKernelRuns;
	cudaError_t error;

	// check number of args
	if(argc != 3){
		printf("try with arguments InputFile and OutputFile");
		return -1;
	}
	// read input file, get inputFileLength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);
	
	// calculate run duration
	start = clock();
	
	//number of input blocks
	unsigned int numInputDataBlocks = (unsigned int)(ceil((float)inputFileLength / BLOCK_SIZE));

	//index of input  blocks
	unsigned int *inputBlocksIndex = (unsigned int *)malloc(numInputDataBlocks * sizeof(unsigned int));

	//allocate frequency
	frequency = (unsigned int **)malloc(numInputDataBlocks * sizeof(unsigned int *));
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		frequency[i] = (unsigned int *)malloc(256 * sizeof(unsigned int));
	}

	//allocate dictionaries
	huffmanDictionary_t *huffmanDictionary = (huffmanDictionary_t *)malloc(numInputDataBlocks * sizeof(huffmanDictionary_t));

	//generate data offset array
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));

	unsigned char *inputBlockPointer = inputFileData;
	unsigned int processLength = inputFileLength;
	unsigned int count = 0;
  while(processLength != 0){
  	unsigned int inputBlockLength = processLength > BLOCK_SIZE ? BLOCK_SIZE : processLength;
	  processLength -= inputBlockLength;

	  //copy input data to global memory
	  memcpy(inputBlockData, inputBlockPointer, inputBlockLength);
		inputBlockPointer += inputBlockLength;

		//initialize frequency and find freq. of each symbol
		intitialize_frequency(frequency[count], inputBlockLength, inputBlockData);

		// initialize nodes of huffman tree
		huffmanTree_t huffmanTreeNode[512];
		unsigned int distinctCharacterCount = intitialize_huffman_tree_get_distinct_char_count(frequency[count], huffmanTreeNode);
	
		// build tree 
		huffmanTree_t *head_huffmanTreeNode = NULL;
		for (unsigned int i = 0; i < distinctCharacterCount - 1; i++){
			unsigned int combinedHuffmanNodes = 2 * i;
			sort_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode);
			build_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode, &head_huffmanTreeNode);
		}
	
		if(distinctCharacterCount == 1){
		head_huffmanTreeNode = &huffmanTreeNode[0];
		}

		// build table having the bitSequence sequence and its length
		unsigned char bitSequence[255], bitSequenceLength = 0;
		build_huffman_dictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength, &huffmanDictionary[count]);
		create_data_offset_array_single_run((inputBlockPointer - inputFileData - inputBlockLength), compressedDataOffset, inputBlockData, inputBlockLength, &huffmanDictionary[count]);
		inputBlocksIndex[count] = compressedDataOffset[inputBlockPointer - inputFileData];
		count++;
	}

	compressedData = (unsigned char *)malloc(sizeof(unsigned char) * (compressedDataOffset[inputFileLength] / 8));

	//gpu memory allocation
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset, *d_inputBlocksIndex;
	huffmanDictionary_t *d_huffmanDictionary;
	
	// allocate memory for input data, offset information and dictionary
	error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_1: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_2: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t));
	if (error != cudaSuccess)
		printf("erro_3: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_inputBlocksIndex, numInputDataBlocks * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_4: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_9: %s\n", cudaGetErrorString(error));


	// memory copy input data, offset information and dictionary
	error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_5: %s\n", cudaGetErrorString(error));
	printf("malloc and copies done 1\n");
	error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_6: %s\n", cudaGetErrorString(error));
	printf("malloc and copies done 2\n");
	error = cudaMemcpy(d_huffmanDictionary, huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_7: %s\n", cudaGetErrorString(error));
	printf("malloc and copies done 3\n");
	error = cudaMemcpy(d_inputBlocksIndex, inputBlocksIndex, numInputDataBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_8: %s\n", cudaGetErrorString(error));
	printf("malloc and copies done 4\n");
		

	// initialize d_byteCompressedData 
	error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_10: %s\n", cudaGetErrorString(error));
	
	printf("before kernel\n");
	
	// run kernel
	compress<<<4, 1024>>>(d_inputBlocksIndex, d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks);
	cudaError_t error_kernel = cudaGetLastError();
	if (error_kernel != cudaSuccess)
		printf("erro_final: %s\n", cudaGetErrorString(error_kernel));

	// copy compressed data from GPU to CPU memory
	error = cudaMemcpy(compressedData, d_inputFileData, ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));
			
	// free allocated memory
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_huffmanDictionary);
	cudaFree(d_byteCompressedData);
	cudaFree(d_inputBlocksIndex);

	// calculate run duration
	end = clock();
	// write src inputFileLength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	
	unsigned char *putputDataPtr = compressedData;
	inputBlockLength = BLOCK_SIZE;
	for(unsigned int i = 0; i < numInputDataBlocks - 1; i++){
		unsigned int compressedBlockLength = (inputBlocksIndex[i + 1] - inputBlocksIndex[i]) / 8;
		fwrite(&compressedBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, compressedFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, compressedFile);
		putputDataPtr += compressedBlockLength;
	}


	compressedBlockLength = (compressedDataOffset[inputFileLength] - inputBlocksIndex[numInputDataBlocks - 1]) / 8;
	fwrite(&compressedBlockLength, sizeof(unsigned int), 1, compressedFile);
	inputBlockLength = inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE;
	fwrite(&inputBlockLength, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency[numInputDataBlocks - 1], sizeof(unsigned int), 256, compressedFile);
	fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, compressedFile);
	fclose(compressedFile);	
	
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedDataOffset);
	free(compressedData);
	return 0;
}
