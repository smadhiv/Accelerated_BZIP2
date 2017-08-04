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
	unsigned int **frequency, inputBlockLength;
	unsigned int inputFileLength;
	unsigned char *inputFileData, *compressedData;

	FILE *inputFile, *compressedFile;

	//gpu specific
	unsigned int *compressedDataOffset;

	// check number of args
	if(argc != 3){
		printf("try with arguments InputFile and OutputFile");
		return -1;
	}
	
	// calculate run duration
	start = clock();

	// read input file, get inputFileLength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);

	//number of input blocks
	unsigned int numInputDataBlocks = (unsigned int)(ceil((float)inputFileLength / BLOCK_SIZE));
	printf("Number of blocks : %u\n", numInputDataBlocks);

	//compute minimum memory req. get GPU memory
	long unsigned int gpuMemoryRequired = 5 * inputFileLength * sizeof(unsigned char) + numInputDataBlocks * sizeof(huffmanDictionary_t) + (int)((float)inputFileLength/10) + 10 * 1024 * 1024;
	long unsigned int mem_free, mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);
	if(mem_free < gpuMemoryRequired){
		printf("Insufficient GPU memory\n");
		return -1;
	}

	//allocate frequency, dictionaries, offset array and index of input  blocks
	frequency = (unsigned int **)malloc(numInputDataBlocks * sizeof(unsigned int *));
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		frequency[i] = (unsigned int *)malloc(256 * sizeof(unsigned int));
	}
	huffmanDictionary_t *huffmanDictionary = (huffmanDictionary_t *)malloc(numInputDataBlocks * sizeof(huffmanDictionary_t));
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));
	unsigned int *inputBlocksIndex = (unsigned int *)malloc((numInputDataBlocks + 1) * sizeof(unsigned int));
	inputBlocksIndex[0] = 0;

	//process input file
	unsigned char *inputBlockPointer = inputFileData;
	unsigned int processLength = inputFileLength;
	unsigned int currentBlockIndex = 0;
	unsigned int integerOverFlowIndex[10];
	unsigned int numIntegerOverflows = 0;
  while(processLength != 0){
  	unsigned int inputBlockLength = processLength > BLOCK_SIZE ? BLOCK_SIZE : processLength;
	  processLength -= inputBlockLength;

	  //copy input data to global memory
	  memcpy(inputBlockData, inputBlockPointer, inputBlockLength);
		inputBlockPointer += inputBlockLength;

		//initialize frequency and find freq. of each symbol
		intitialize_frequency(frequency[currentBlockIndex], inputBlockLength, inputBlockData);

		// initialize nodes of huffman tree
		huffmanTree_t huffmanTreeNode[512];
		unsigned int distinctCharacterCount = intitialize_huffman_tree_get_distinct_char_count(frequency[currentBlockIndex], huffmanTreeNode);
	
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
		build_huffman_dictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength, &huffmanDictionary[currentBlockIndex]);
		create_data_offset_array((inputBlockPointer - inputFileData - inputBlockLength), compressedDataOffset, inputBlockData, inputBlockLength, &huffmanDictionary[currentBlockIndex], integerOverFlowIndex, &numIntegerOverflows);
		inputBlocksIndex[currentBlockIndex + 1] = compressedDataOffset[inputBlockPointer - inputFileData];
		currentBlockIndex++;
	}
	
	unsigned int compressedFileLength = compressedDataOffset[inputFileLength] / 8;
	for(unsigned int i = 0; i < numIntegerOverflows; i++){
		compressedFileLength += (compressedDataOffset[integerOverFlowIndex[i]] / 8);
	}
	compressedData = (unsigned char *)malloc(sizeof(unsigned char) * (compressedFileLength));

	//gpu memory allocation
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	cudaError_t error;
	huffmanDictionary_t *d_huffmanDictionary;
	
	// allocate memory for input data, offset information and dictionary
	unsigned int gpuInputDataAllocationLength = inputFileLength >= compressedFileLength ? inputFileLength : compressedFileLength;
	error = cudaMalloc((void **)&d_inputFileData, gpuInputDataAllocationLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_input_data: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_data_offset: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t));
	if (error != cudaSuccess)
		printf("erro_dictionary: %s\n", cudaGetErrorString(error));

	// memory copy input data, offset information and dictionary
	error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_input_data_mem: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_data_offset_mem: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_huffmanDictionary, huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_dictionary_mem: %s\n", cudaGetErrorString(error));
	
	// run kernel
	if(numIntegerOverflows == 0){
		printf("No Overflow!!\n");

		error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
		error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		encode_single_run_no_overflow<<<4, 1024>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks);
		compress_single_run_no_overflow<<<4, 1024>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, inputFileLength);
	}
	else{
		printf("With Overflow!!\n");
		unsigned char *d_byteCompressedData_overflow;

		error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[integerOverFlowIndex[0]]) * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
		error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverFlowIndex[0]] * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		error = cudaMalloc((void **)&d_byteCompressedData_overflow, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
		error = cudaMemset(d_byteCompressedData_overflow, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		encode_single_run_with_overflow<<<4, 1024>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks, integerOverFlowIndex[0] / BLOCK_SIZE, d_byteCompressedData_overflow);
		compress_single_run_with_overflow<<<4, 1024>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, inputFileLength, integerOverFlowIndex[0] / BLOCK_SIZE, d_byteCompressedData_overflow);
	}

	cudaError_t error_kernel = cudaGetLastError();
	if (error_kernel != cudaSuccess)
		printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));

	// copy compressed data from GPU to CPU memory
	error = cudaMemcpy(compressedData, d_inputFileData, compressedFileLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));
			
	// calculate run duration
	end = clock();

	// write src inputFileLength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	unsigned char *putputDataPtr = compressedData;
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		//accounting for integeroverflow below
		unsigned int compressedBlockLength = inputBlocksIndex[i + 1] > inputBlocksIndex[i] ? (inputBlocksIndex[i + 1] - inputBlocksIndex[i]) / 8 : inputBlocksIndex[i + 1] / 8;

		inputBlockLength = i != numInputDataBlocks - 1 ? BLOCK_SIZE : (inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE);

		fwrite(&compressedBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, compressedFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, compressedFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, compressedFile);
		putputDataPtr += compressedBlockLength;
	}

	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedDataOffset);
	free(compressedData);
	// free allocated memory
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_huffmanDictionary);
	cudaFree(d_byteCompressedData);
	return 0;
}

	//	unsigned int *d_inputBlocksIndex;
	//error = cudaMalloc((void **)&d_inputBlocksIndex, numInputDataBlocks * sizeof(unsigned int));
	//if (error != cudaSuccess)
	//	printf("erro_4: %s\n", cudaGetErrorString(error));
	//error = cudaMemcpy(d_inputBlocksIndex, inputBlocksIndex, numInputDataBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//if (error!= cudaSuccess)
	//		printf("erro_8: %s\n", cudaGetErrorString(error));	
	//	cudaFree(d_inputBlocksIndex);
