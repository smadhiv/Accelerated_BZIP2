#ifndef huffman_parallel
#include "huffman_parallel.h"
#endif

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int compute_mem_offset(unsigned int *frequency, huffmanDictionary_t* huffmanDictionary){
	// offset array requirements
	long unsigned int mem_offset = 0;
	for(unsigned int i = 0; i < 256; i++){
		mem_offset += frequency[i] * (*huffmanDictionary).bitSequenceLength[i];
	}
	mem_offset = mem_offset % 8 == 0 ? mem_offset : mem_offset + 8 - mem_offset % 8;
	return mem_offset;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//initialize frequency array with histogram of input data
void intitialize_frequency(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData){
	//compute frequency of input characters
	for (unsigned int i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (unsigned int i = 0; i < inputBlockLength; i++){
		frequency[inputBlockData[i]]++;
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//intitialize huffmantree nodes with the character and its frequency
//returns the number of distinct values in the given input data
unsigned int intitialize_huffman_tree_get_distinct_char_count(unsigned int *frequency, huffmanTree_t *huffmanTreeNode){
	//initialize nodes of huffman tree
	unsigned int distinctCharacterCount = 0;
	for (unsigned int i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacterCount].count = frequency[i];
			huffmanTreeNode[distinctCharacterCount].letter = i;
			huffmanTreeNode[distinctCharacterCount].left = NULL;
			huffmanTreeNode[distinctCharacterCount].right = NULL;
			distinctCharacterCount++;
		}
	}
	return distinctCharacterCount;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// sort huffmantree nodes based on frequency
void sort_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int mergedHuffmanNodes, huffmanTree_t *huffmanTreeNode){
	unsigned int a, b;
	for (a = mergedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = mergedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				huffmanTree_t temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// build tree based on the above sort result
void build_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int mergedHuffmanNodes, huffmanTree_t *huffmanTreeNode, huffmanTree_t **head_huffmanTreeNode){
	huffmanTreeNode[distinctCharacterCount + i].count = huffmanTreeNode[mergedHuffmanNodes].count + huffmanTreeNode[mergedHuffmanNodes + 1].count;
	huffmanTreeNode[distinctCharacterCount + i].left = &huffmanTreeNode[mergedHuffmanNodes];
	huffmanTreeNode[distinctCharacterCount + i].right = &huffmanTreeNode[mergedHuffmanNodes + 1];
	*head_huffmanTreeNode = &(huffmanTreeNode[distinctCharacterCount + i]);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// get bitSequence sequence for each character value
void build_huffman_dictionary(huffmanTree_t *root, unsigned char *bitSequence, unsigned char bitSequenceLength, 	huffmanDictionary_t *huffmanDictionary){
	if (root->left){
		bitSequence[bitSequenceLength] = 0;
		build_huffman_dictionary(root->left, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->right){
		bitSequence[bitSequenceLength] = 1;
		build_huffman_dictionary(root->right, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->left == NULL && root->right == NULL){
		(*huffmanDictionary).bitSequenceLength[root->letter] = bitSequenceLength;
		if(bitSequenceLength < 192){
			memcpy((*huffmanDictionary).bitSequence[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
		}
		else{
			printf("dictioanary length exceed 192 ;(\n");
			//memcpy(bitSequenceConstMemory[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
			//memcpy((*huffmanDictionary).bitSequence[root->letter], bitSequence, 191);
			//constMemoryFlag = 1;
		}
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//builds the output data 
unsigned int generate_compressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *compressedBlockData, huffmanDictionary_t *huffmanDictionary){
	unsigned char writeBit = 0, bitsFilled = 0;
	unsigned int compressedBlockLength = 0;

	for (unsigned int i = 0; i < inputBlockLength; i++){
		for (unsigned char j = 0; j < *huffmanDictionary[inputBlockData[i]].bitSequenceLength; j++){
			if (huffmanDictionary[inputBlockData[i]].bitSequence[j] == 0){
				writeBit = writeBit << 1;
				bitsFilled++;
			}
			else{
				writeBit = (writeBit << 1) | 01;
				bitsFilled++;
			}
			if (bitsFilled == 8){
				compressedBlockData[compressedBlockLength] = writeBit;
				bitsFilled = 0;
				writeBit = 0;
				compressedBlockLength++;
			}
		}
	}

	if (bitsFilled != 0){
		for (unsigned int i = 0; (unsigned char)i < 8 - bitsFilled; i++){
			writeBit = writeBit << 1;
		}
		compressedBlockData[compressedBlockLength] = writeBit;
		compressedBlockLength++;
	}
	return compressedBlockLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// the function calls above functions to generate compressed data
//returns the size of compressed data
unsigned int huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* compressedBlockData){
	intitialize_frequency(frequency, inputBlockLength, inputBlockData);

	huffmanTree_t huffmanTreeNode[512];
	unsigned int distinctCharacterCount = intitialize_huffman_tree_get_distinct_char_count(frequency, huffmanTreeNode);

	// build tree 
	huffmanTree_t *head_huffmanTreeNode = NULL;
	for (unsigned int i = 0; i < distinctCharacterCount - 1; i++){
		unsigned int combinedHuffmanNodes = 2 * i;
		sort_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode);
		build_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode, &head_huffmanTreeNode);
	}
	
	// build table having the bitSequence sequence and its length
	huffmanDictionary_t huffmanDictionary[256];
	unsigned char bitSequence[255], bitSequenceLength = 0;
	build_huffman_dictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength, 	huffmanDictionary);

	// compress
	unsigned int compressedBlockLength = generate_compressed_data(inputBlockLength, inputBlockData, compressedBlockData, huffmanDictionary);
	return compressedBlockLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//create offset array to write bit sequence
void create_data_offset_array(int index, unsigned int *compressedDataOffset, unsigned char* inputBlockData, unsigned int inputBlockLength, huffmanDictionary_t *huffmanDictionary, unsigned int *integerOverFlowBlockIndex, unsigned int *numIntegerOverflows, unsigned int *kernelOverFlowIndex, unsigned int *numKernelRuns, long unsigned int *mem_used, long unsigned int mem_avail){
	
	compressedDataOffset[0] = 0;
	unsigned int *dataOffsetIndex = compressedDataOffset + index;
	unsigned int i = 0;
	while(i < inputBlockLength){
		dataOffsetIndex[i + 1] = (*huffmanDictionary).bitSequenceLength[inputBlockData[i]] + dataOffsetIndex[i];
		if((*mem_used) + dataOffsetIndex[i + 1] > mem_avail){
			kernelOverFlowIndex[(*numKernelRuns)] = index;
			(*numKernelRuns)++;
			dataOffsetIndex[1] = (*huffmanDictionary).bitSequenceLength[inputBlockData[0]];
			i = 0;
			*mem_used = 0;
		}
		else if (dataOffsetIndex[i + 1] + 16 * 1024 < dataOffsetIndex[i]){
			integerOverFlowBlockIndex[(*numIntegerOverflows)] = index;
			(*numIntegerOverflows)++;
			dataOffsetIndex[1] = (*huffmanDictionary).bitSequenceLength[inputBlockData[0]];
			i = 0;
			*mem_used = dataOffsetIndex[0];
		}
		i++;
	}
	if(dataOffsetIndex[inputBlockLength] % 8 != 0){
		dataOffsetIndex[inputBlockLength] = dataOffsetIndex[inputBlockLength] + (8 - (dataOffsetIndex[inputBlockLength] % 8));
	}		
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int build_compressed_data_offset(unsigned int *compressedDataOffset, unsigned int *inputBlocksIndex, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, unsigned char *inputBlockData, unsigned int inputFileLength, unsigned char *inputFileData, unsigned int *numIntegerOverflows,  unsigned int *integerOverFlowIndex, unsigned int *numKernelRuns,  unsigned int *kernelOverFlowIndex, long unsigned int mem_avail){
//process input file
	unsigned int currentBlockIndex = 0;
	long unsigned int mem_used = 0;
	unsigned char *inputBlockPointer = inputFileData;
	unsigned int processLength = inputFileLength;
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
		create_data_offset_array((inputBlockPointer - inputFileData - inputBlockLength), compressedDataOffset, inputBlockData, inputBlockLength, &huffmanDictionary[currentBlockIndex], integerOverFlowIndex, numIntegerOverflows, kernelOverFlowIndex, numKernelRuns, &mem_used, mem_avail);
		inputBlocksIndex[currentBlockIndex + 1] = compressedDataOffset[inputBlockPointer - inputFileData];
		currentBlockIndex++;
		if(numKernelRuns < numIntegerOverflows || (*numKernelRuns) > 9 || (*numIntegerOverflows) > 9){
			return -1;
		}
	}
	return 0;
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void cuda_compress_single_run_no_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength){
	printf("No Overflow and single run!!\n");

	//gpu memory allocation
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	huffmanDictionary_t *d_huffmanDictionary;
	cudaError_t error;

	//compressed data
	unsigned int compressedFileLength = compressedDataOffset[inputFileLength] / 8;
	unsigned char *compressedData = (unsigned char *)malloc(compressedFileLength * sizeof(unsigned char));

	// allocate memory for input data, offset information and dictionary
	unsigned int gpuInputDataAllocationLength = inputFileLength > compressedFileLength ? inputFileLength : compressedFileLength;
	error = cudaMalloc((void **)&d_inputFileData, gpuInputDataAllocationLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_input_data: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_data_offset: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
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
	error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_memset: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_huffmanDictionary, huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_dictionary_mem: %s\n", cudaGetErrorString(error));

	//call kernel
	encode_single_run_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks);
	compress_single_run_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, inputFileLength);
	cudaError_t error_kernel = cudaGetLastError();
	if (error_kernel != cudaSuccess)
		printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));

	// copy compressed data from GPU to CPU memory
	error = cudaMemcpy(compressedData, d_inputFileData, compressedFileLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));

	//write to output file
	FILE *outputFile = fopen(outputFileName, "wb");
	unsigned char *putputDataPtr = compressedData;
	for(unsigned int i = 0; i < numInputDataBlocks; i++){

		//i/o lengths
		unsigned int compressedBlockLength = arrayCompressedBlocksLength[i];
		unsigned int inputBlockLength = i != numInputDataBlocks - 1 ? BLOCK_SIZE : (inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE);

		//writes
		fwrite(&arrayCompressedBlocksLength[i], sizeof(unsigned int), 1, outputFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, outputFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, outputFile);
		putputDataPtr += compressedBlockLength;
	}

	// free allocated memory
	free(outputFile);
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_huffmanDictionary);
	cudaFree(d_byteCompressedData);
	fclose(outputFile);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void cuda_compress_single_run_with_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int integerOverFlowIndex){
	printf("No Overflow and single run!!\n");

	//gpu memory allocation
	unsigned char *d_inputFileData;
	unsigned char *d_byteCompressedData_overflow, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	huffmanDictionary_t *d_huffmanDictionary;
	cudaError_t error;

	//compressed data
	unsigned int compressedFileLength = (compressedDataOffset[integerOverFlowIndex] / 8) + (compressedDataOffset[inputFileLength] / 8);
	unsigned char *compressedData = (unsigned char *)malloc(compressedFileLength * sizeof(unsigned char));

	// allocate memory for input data, offset information and dictionary
	unsigned int gpuInputDataAllocationLength = inputFileLength > compressedFileLength ? inputFileLength : compressedFileLength;
	error = cudaMalloc((void **)&d_inputFileData, gpuInputDataAllocationLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_input_data: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_data_offset: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[integerOverFlowIndex]) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData_overflow, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
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
	error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverFlowIndex] * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_memset: %s\n", cudaGetErrorString(error));
	error = cudaMemset(d_byteCompressedData_overflow, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_memset: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_huffmanDictionary, huffmanDictionary, numInputDataBlocks * sizeof(huffmanDictionary_t), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_dictionary_mem: %s\n", cudaGetErrorString(error));

	//call kernel
	encode_single_run_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks, integerOverFlowIndex / BLOCK_SIZE, d_byteCompressedData_overflow);
	compress_single_run_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, inputFileLength, integerOverFlowIndex / BLOCK_SIZE, d_byteCompressedData_overflow);
	cudaError_t error_kernel = cudaGetLastError();
	if (error_kernel != cudaSuccess)
		printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));

		// copy compressed data from GPU to CPU memory
		error = cudaMemcpy(compressedData, d_inputFileData, compressedFileLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));

	//write to output file
	FILE *outputFile = fopen(outputFileName, "wb");
	unsigned char *putputDataPtr = compressedData;
	for(unsigned int i = 0; i < numInputDataBlocks; i++){

		//i/o lengths
		unsigned int compressedBlockLength = arrayCompressedBlocksLength[i];
		unsigned int inputBlockLength = i != numInputDataBlocks - 1 ? BLOCK_SIZE : (inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE);

		//writes
		fwrite(&arrayCompressedBlocksLength[i], sizeof(unsigned int), 1, outputFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, outputFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, outputFile);
		putputDataPtr += compressedBlockLength;
	}

	// free allocated memory
	fclose(outputFile);
	free(compressedData);
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_byteCompressedData);
	cudaFree(d_byteCompressedData_overflow);
	cudaFree(d_huffmanDictionary);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void cuda_compress_multiple_run_no_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int numKernelRuns, unsigned int *gpuOverFlowIndex){
	printf("No Overflow and multiple run!!\n");

	//gpu memory allocation
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	huffmanDictionary_t *d_huffmanDictionary;
	cudaError_t error;

	//compressed data
	unsigned int compressedFileLength = compressedDataOffset[inputFileLength] / 8;
	unsigned int maxByteStreamLength = compressedDataOffset[inputFileLength];
	for(unsigned int i = 0; i < numKernelRuns; i++){
		maxByteStreamLength = maxByteStreamLength > compressedDataOffset[gpuOverFlowIndex[i]] ? maxByteStreamLength : compressedDataOffset[gpuOverFlowIndex[i]];
		compressedFileLength += compressedDataOffset[gpuOverFlowIndex[i]] / 8;
	}
	unsigned char *compressedData = (unsigned char *)malloc(compressedFileLength * sizeof(unsigned char));

	// allocate memory for input data, offset information and dictionary
	unsigned int gpuInputDataAllocationLength = inputFileLength > compressedFileLength ? inputFileLength : compressedFileLength;
	error = cudaMalloc((void **)&d_inputFileData, gpuInputDataAllocationLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_input_data: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_data_offset: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData, (maxByteStreamLength) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
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

	//call kernel
	unsigned int writePosition = 0;
	for(int i = 0; i < numKernelRuns; i++){
		//memset
		error = cudaMemset(d_byteCompressedData, 0, maxByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		//launch
		encode_multiple_runs_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks, gpuOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, inputFileLength);
		compress_multiple_runs_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, gpuOverFlowIndex[i + 1], writePosition);
		cudaError_t error_kernel = cudaGetLastError();
		if (error_kernel != cudaSuccess)
			printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));

		//write position
		writePosition += compressedDataOffset[gpuOverFlowIndex[i]] / 8;
	}

	// copy compressed data from GPU to CPU memory
	error = cudaMemcpy(compressedData, d_inputFileData, compressedFileLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));

	//write to output file
	FILE *outputFile = fopen(outputFileName, "wb");
	unsigned char *putputDataPtr = compressedData;
	for(unsigned int i = 0; i < numInputDataBlocks; i++){

		//i/o lengths
		unsigned int compressedBlockLength = arrayCompressedBlocksLength[i];
		unsigned int inputBlockLength = i != numInputDataBlocks - 1 ? BLOCK_SIZE : (inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE);

		//writes
		fwrite(&arrayCompressedBlocksLength[i], sizeof(unsigned int), 1, outputFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, outputFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, outputFile);
		putputDataPtr += compressedBlockLength;
	}

	// free allocated memory
	free(outputFile);
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_huffmanDictionary);
	cudaFree(d_byteCompressedData);
	fclose(outputFile);
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void cuda_compress_multiple_run_with_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int numKernelRuns, unsigned int *gpuOverFlowIndex, unsigned int numIntegerOverflows, unsigned int *integerOverFlowIndex){
	printf("with Overflow and multiple run!!\n");

	//gpu memory allocation
	unsigned char *d_inputFileData;
	unsigned char *d_byteCompressedData_overflow, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	huffmanDictionary_t *d_huffmanDictionary;
	cudaError_t error;

	//compressed data
	unsigned int compressedFileLength = compressedDataOffset[inputFileLength] / 8;
	unsigned int maxOverFlowByteStreamLength = compressedDataOffset[inputFileLength];
	for(unsigned int i = 0; i < numKernelRuns; i++){
		maxOverFlowByteStreamLength = maxOverFlowByteStreamLength > compressedDataOffset[gpuOverFlowIndex[i]] ? maxOverFlowByteStreamLength : compressedDataOffset[gpuOverFlowIndex[i]];
		compressedFileLength += compressedDataOffset[gpuOverFlowIndex[i]] / 8;
	}
	unsigned int maxByteStreamLength = 0;
	for(unsigned int i = 0; i < numIntegerOverflows; i++){
		maxByteStreamLength = maxByteStreamLength > compressedDataOffset[integerOverFlowIndex[i]] ? maxByteStreamLength : compressedDataOffset[integerOverFlowIndex[i]];
		compressedFileLength += compressedDataOffset[integerOverFlowIndex[i]] / 8;
	}

	unsigned char *compressedData = (unsigned char *)malloc(compressedFileLength * sizeof(unsigned char));

	// allocate memory for input data, offset information and dictionary
	unsigned int gpuInputDataAllocationLength = inputFileLength > compressedFileLength ? inputFileLength : compressedFileLength;
	error = cudaMalloc((void **)&d_inputFileData, gpuInputDataAllocationLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_input_data: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_data_offset: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData, (maxByteStreamLength) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_byteCompressedData_overflow, (maxOverFlowByteStreamLength) * sizeof(unsigned char));
	if (error!= cudaSuccess)
		printf("erro_byte_compressed: %s\n", cudaGetErrorString(error));
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

	//call kernel
	unsigned int writePosition = 0;
	for(int i = 0; i < numKernelRuns - 1; i++){
		//memset
		error = cudaMemset(d_byteCompressedData, 0, maxByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));
		error = cudaMemset(d_byteCompressedData_overflow, 0, maxOverFlowByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		//launch
		encode_multiple_runs_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, compressedDataOffset[gpuOverFlowIndex[i + 1]], numInputDataBlocks, gpuOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, integerOverFlowIndex[i] / BLOCK_SIZE, d_byteCompressedData_overflow, inputFileLength);
		compress_multiple_runs_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, compressedDataOffset[gpuOverFlowIndex[i + 1]], writePosition, integerOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, d_byteCompressedData_overflow);
		cudaError_t error_kernel = cudaGetLastError();
		if (error_kernel != cudaSuccess)
			printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));

		//write position
		writePosition += compressedDataOffset[gpuOverFlowIndex[i]] / 8 + compressedDataOffset[gpuOverFlowIndex[i]] / 8;
	}

	if(numKernelRuns == numIntegerOverflows){
		unsigned int i = numKernelRuns - 1;
		//memset
		error = cudaMemset(d_byteCompressedData, 0, maxByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));
		error = cudaMemset(d_byteCompressedData_overflow, 0, maxOverFlowByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		//launch
		encode_multiple_runs_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, compressedDataOffset[gpuOverFlowIndex[i + 1]], numInputDataBlocks, gpuOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, integerOverFlowIndex[i] / BLOCK_SIZE, d_byteCompressedData_overflow, inputFileLength);
		compress_multiple_runs_with_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, compressedDataOffset[gpuOverFlowIndex[i + 1]], writePosition, integerOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, d_byteCompressedData_overflow);
		cudaError_t error_kernel = cudaGetLastError();
		if (error_kernel != cudaSuccess)
			printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));		
	}
	else{
		//memset
		unsigned int i = numKernelRuns - 1;
		error = cudaMemset(d_byteCompressedData, 0, maxByteStreamLength * sizeof(unsigned char));
		if (error!= cudaSuccess)
			printf("erro_memset: %s\n", cudaGetErrorString(error));

		//launch
		encode_multiple_runs_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, numInputDataBlocks, gpuOverFlowIndex[i] / BLOCK_SIZE, gpuOverFlowIndex[i + 1] / BLOCK_SIZE, inputFileLength);
		compress_multiple_runs_no_overflow<<<GRID_DIM, BLOCK_DIM>>>(d_inputFileData, d_compressedDataOffset, d_byteCompressedData, gpuOverFlowIndex[i + 1], writePosition);
		cudaError_t error_kernel = cudaGetLastError();
		if (error_kernel != cudaSuccess)
			printf("erro_kernel: %s\n", cudaGetErrorString(error_kernel));
	}

	// copy compressed data from GPU to CPU memory
	error = cudaMemcpy(compressedData, d_inputFileData, compressedFileLength * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("erro_copy_compressed_data: %s\n", cudaGetErrorString(error));

	//write to output file
	FILE *outputFile = fopen(outputFileName, "wb");
	unsigned char *putputDataPtr = compressedData;
	for(unsigned int i = 0; i < numInputDataBlocks; i++){

		//i/o lengths
		unsigned int compressedBlockLength = arrayCompressedBlocksLength[i];
		unsigned int inputBlockLength = i != numInputDataBlocks - 1 ? BLOCK_SIZE : (inputFileLength % BLOCK_SIZE != 0 ? inputFileLength % BLOCK_SIZE : BLOCK_SIZE);

		//writes
		fwrite(&arrayCompressedBlocksLength[i], sizeof(unsigned int), 1, outputFile);
		fwrite(&inputBlockLength, sizeof(unsigned int), 1, outputFile);
		fwrite(frequency[i], sizeof(unsigned int), 256, outputFile);
		fwrite(putputDataPtr, sizeof(unsigned char), compressedBlockLength, outputFile);
		putputDataPtr += compressedBlockLength;
	}

	// free allocated memory
	free(outputFile);
	cudaFree(d_inputFileData);
	cudaFree(d_compressedDataOffset);
	cudaFree(d_huffmanDictionary);
	cudaFree(d_byteCompressedData);
	cudaFree(d_byteCompressedData_overflow);
	fclose(outputFile);
}