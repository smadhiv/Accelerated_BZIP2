/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include "../../../library/huffman/parallel/huffman_parallel.h"
#include "../../../library/huffman/parallel/parallelHeader.h"

//global variables to be used in qsort function
//extern unsigned int inputBlockLength;
//store input block data
//remove this for cudabzip2
unsigned char inputBlockData[BLOCK_SIZE];
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int **frequency, compressedBlockLength, inputBlockLength;
	unsigned int inputFileLength;
	unsigned char *inputFileData, *compressedData;

	struct huffmanTree *head_huffmanTreeNode;
	struct huffmanTree huffmanTreeNode[512];

	FILE *inputFile, *compressedFile;

	//gpu specific
	unsigned int *compressedDataOffset;
	unsigned int integerOverflowFlag;
	long unsigned int mem_req;
	int numKernelRuns;

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
	for(unsigned int i = 0; i < numInputDataBlocks; di++){
		frequency[i] = (int *)malloc(256 * sizeof(unsigned int));
	}

	//allocate dictionaries
	struct huffmanDictionary *huffmanDictionary = (struct huffmanDictionary *)malloc(numInputDataBlocks * sizeof(struct huffmanDictionary));

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
		struct huffmanTree huffmanTreeNode[512];
		unsigned int distinctCharacterCount = intitialize_huffman_tree_get_distinct_char_count(frequency, huffmanTreeNode);
	
		// build tree 
		struct huffmanTree *head_huffmanTreeNode = NULL;
		for (unsigned int i = 0; i < distinctCharacterCount - 1; i++){
			unsigned int combinedHuffmanNodes = 2 * i;
			sort_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode);
			build_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode, &head_huffmanTreeNode);
		}
	
		// build table having the bitSequence sequence and its length
		unsigned char bitSequence[255], bitSequenceLength = 0;
		build_huffman_dictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength, huffmanDictionary[count]);
		create_data_offset_array((inputBlockPointer - inputFileData), compressedDataOffset, inputBlockData, inputBlockLength, huffmanDictionary[count]);
		inputBlocksIndex[inputBlockPointer - inputFileData] = compressedDataOffset[inputBlockPointer - inputFileData];
		count++;
	}

	//gpu memory allocation
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	struct huffmanDictionary *d_huffmanDictionary;
	
	// allocate memory for input data, offset information and dictionary
	error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
	if (error != cudaSuccess)
		printf("erro_1: %s\n", cudaGetErrorString(error));
		
	error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
		printf("erro_2: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_huffmanDictionary, numInputDataBlocks * sizeof(struct huffmanDictionary));
	if (error != cudaSuccess)
		printf("erro_3: %s\n", cudaGetErrorString(error));

		// memory copy input data, offset information and dictionary
		error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_4: %s\n", cudaGetErrorString(error));
		error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_5: %s\n", cudaGetErrorString(error));
		error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, numInputDataBlocks * sizeof(struct huffmanDictionary), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_6: %s\n", cudaGetErrorString(error));
			
		// copy constant memory if required for dictionary
		if(constMemoryFlag == 1){
			error = cudaMemcpyToSymbol (d_bitSequenceConstMemory, bitSequenceConstMemory, 256 * 255 * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("erro_const: %s\n", cudaGetErrorString(error));
		}


	// calculate memory requirements
	long unsigned int mem_offset = compute_mem_offset(frequency, &huffmanDictionary);
  long unsigned int mem_data = inputFileLength + (inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);

	if(mem_free - mem_data < MIN_SCRATCH_SIZE){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
		return -1;
	}

	// GPU memory
	long unsigned int mem_free, mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);
	// debug
	if(1){
		printf("Free Mem: %lu\n", mem_free);		
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
