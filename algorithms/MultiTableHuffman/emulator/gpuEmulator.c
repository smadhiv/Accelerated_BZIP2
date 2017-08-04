/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#ifndef huffman_gpu_emulation
#include "../../../library/huffman/gpu_emulation/huffman_gpu_emulation.h"
#endif

//global variables to be used in qsort function
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

	//allocate frequency, dictionaries, offset array and index of input  blocks
	frequency = (unsigned int **)malloc(numInputDataBlocks * sizeof(unsigned int *));
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		frequency[i] = (unsigned int *)malloc(256 * sizeof(unsigned int));
	}
	huffmanDictionary_t *huffmanDictionary = (huffmanDictionary_t *)malloc(numInputDataBlocks * sizeof(huffmanDictionary_t));
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));
	unsigned int *inputBlocksIndex = (unsigned int *)malloc((numInputDataBlocks + 1) * sizeof(unsigned int));
	inputBlocksIndex[0] = 0;

	unsigned int flag = 0;
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
		
		if(numIntegerOverflows == 1 && flag == 0){
			flag = 1;
			printf("before overflow %u\n", inputBlocksIndex[currentBlockIndex]);
			printf("after overflow %u\n", inputBlocksIndex[currentBlockIndex + 1]);
			printf("this should be before %u\n", integerOverFlowIndex[0]);
		}
		currentBlockIndex++;
	}
	
	unsigned int compressedFileLength = compressedDataOffset[inputFileLength] / 8;
	for(unsigned int i = 0; i < numIntegerOverflows; i++){
		compressedFileLength += (compressedDataOffset[integerOverFlowIndex[i]] / 8);
		printf("before overflow %u\n", compressedDataOffset[integerOverFlowIndex[i]]);
	}
	compressedData = (unsigned char *)malloc(sizeof(unsigned char) * (compressedFileLength));
	printf("compressed block length = %u\n", compressedFileLength);

	// calculate run duration
	end = clock();

	
	if(numIntegerOverflows == 0){
		printf("No integer overflow!!\n");
		unsigned char *byteCompressedData = (unsigned char *)calloc(compressedDataOffset[inputFileLength], sizeof(unsigned char));
		//encode
		for(unsigned int i = 0; i < numInputDataBlocks; i++){
    	unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  	for(unsigned int j = (i * BLOCK_SIZE); j < upperLimit; j++){
		  	for(unsigned int k = 0; k < huffmanDictionary[i].bitSequenceLength[inputFileData[j]]; k++){
			  	byteCompressedData[compressedDataOffset[j] + k] = huffmanDictionary[i].bitSequence[inputFileData[j]][k];
		  	}
	  	}
  	}
		//compress
		unsigned int upperLimit = compressedDataOffset[inputFileLength];
 		for(unsigned int i = 0; i < upperLimit; i += 8){
	  	for(unsigned int j = 0; j < 8; j++){
		  	if(byteCompressedData[i + j] == 0){
			  	compressedData[i / 8] = compressedData[i / 8] << 1;
		  	}
		  	else{
			  	compressedData[i / 8] = (compressedData[i / 8] << 1) | 1;
		  	}
	  	}
  	}
	}
	
	//integer overflow
	else{
		printf("Integer overflow!!\n");
		unsigned int overFlowBlock = integerOverFlowIndex[0] / BLOCK_SIZE;
		unsigned char *byteCompressedData = calloc(compressedDataOffset[integerOverFlowIndex[0]], sizeof(unsigned char));
		unsigned char *byteCompressedData_overflow = calloc(compressedDataOffset[inputFileLength], sizeof(unsigned char));

		for(unsigned int i = 0; i < overFlowBlock; i++){

			//encode
    	unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  	for(unsigned int j = (i * BLOCK_SIZE); j < upperLimit; j++){
		  	for(unsigned int k = 0; k < huffmanDictionary[i].bitSequenceLength[inputFileData[j]]; k++){
			  	byteCompressedData[compressedDataOffset[j] + k] = huffmanDictionary[i].bitSequence[inputFileData[j]][k];
		  	}
	  	}
  	}

		for(unsigned int i = overFlowBlock; i < numInputDataBlocks; i++){
			printf("im here \n");
    	unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;
	  	for(unsigned int j = (i * BLOCK_SIZE); j < upperLimit; j++){
				if(i == overFlowBlock && j == (i * BLOCK_SIZE)){
		  		for(unsigned int k = 0; k < huffmanDictionary[i].bitSequenceLength[inputFileData[j]]; k++){
			  		byteCompressedData_overflow[k] = huffmanDictionary[i].bitSequence[inputFileData[j]][k];
		  		}					
					continue;
				}
		  	for(unsigned int k = 0; k < huffmanDictionary[i].bitSequenceLength[inputFileData[j]]; k++){
			  	byteCompressedData_overflow[compressedDataOffset[j] + k] = huffmanDictionary[i].bitSequence[inputFileData[j]][k];
		  	}
	  	}
  	}
		printf("encode done\n");
		//return 0;
		//compress
		unsigned int upperLimit_1 = compressedDataOffset[overFlowBlock * BLOCK_SIZE];
		for(unsigned int i = 0; i < upperLimit_1; i += 8){
			for(unsigned int j = 0; j < 8; j++){
				if(byteCompressedData[i + j] == 0){
					compressedData[i / 8] = compressedData[i / 8] << 1;
				}
				else{
					compressedData[i / 8] = (compressedData[i / 8] << 1) | 1;
				}
			}
		}
		printf("compress part 1 done\n");
		
		unsigned int offset_overflow = compressedDataOffset[overFlowBlock * BLOCK_SIZE] / 8;
		unsigned int upperLimit_2 = compressedDataOffset[inputFileLength];
		for(unsigned int i = 0; i < upperLimit_2; i += 8){
			for(unsigned int j = 0; j < 8; j++){
				if(byteCompressedData_overflow[i + j] == 0){
					compressedData[(i / 8) + offset_overflow] = compressedData[(i / 8) + offset_overflow] << 1;
				}
				else{
					compressedData[(i / 8) + offset_overflow] = (compressedData[(i / 8) + offset_overflow] << 1) | 1;
				}
			}
		}
		printf("compress done\n");
	}

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
	return 0;
}
