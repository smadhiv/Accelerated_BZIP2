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

int main(int argc, char **argv){
	//time
	clock_t start, end;
	unsigned int cpu_time_used;

	//files	and data
	FILE *inputFile;
	unsigned char *inputFileData;
	unsigned int inputFileLength;

	//huffman specific
	unsigned int **frequency;
	unsigned int *compressedDataOffset;
	unsigned int numInputDataBlocks;
	huffmanDictionary_t *huffmanDictionary;
	unsigned int *inputBlocksIndex;

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
	numInputDataBlocks = (unsigned int)(ceil((float)inputFileLength / BLOCK_SIZE));
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);

	//info
	printf("Number of blocks : %u\n", numInputDataBlocks);

	//compute minimum memory req. get GPU memory
	long unsigned int gpuMemoryRequired = 5 * inputFileLength * sizeof(unsigned char) + numInputDataBlocks * sizeof(huffmanDictionary_t) + MIN_SCRATCH_SIZE;
	long unsigned int mem_free, mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);
	if(mem_free < gpuMemoryRequired){
		printf("Insufficient GPU memory\n");
		return -1;
	}
	long unsigned int mem_avail = mem_free - gpuMemoryRequired;

	//allocate frequency
	frequency = (unsigned int **)malloc(numInputDataBlocks * sizeof(unsigned int *));
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		frequency[i] = (unsigned int *)malloc(256 * sizeof(unsigned int));
	}

	//allocate dictionary
	huffmanDictionary = (huffmanDictionary_t *)malloc(numInputDataBlocks * sizeof(huffmanDictionary_t));

	//offset 
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));

	//index of each compressed block
	inputBlocksIndex = (unsigned int *)malloc((numInputDataBlocks + 1) * sizeof(unsigned int));
	inputBlocksIndex[0] = 0;

	//build the compressed data offset from input data
	unsigned int numIntegerOverflows = 0, integerOverFlowIndex[10];
	unsigned int numKernelRuns = 1, gpuOverFlowIndex[10];
	gpuOverFlowIndex[0] = 0;
	int ret = build_compressed_data_offset(compressedDataOffset, inputBlocksIndex, huffmanDictionary, frequency, inputBlockData, inputFileLength, inputFileData, &numIntegerOverflows, integerOverFlowIndex, &numKernelRuns, gpuOverFlowIndex, mem_avail);

	if(ret){
		printf("Sorry! have not handled multiple integer overflows in a single kernel execution\nExiting\n");
		return -1;
	}
	else{
		FILE *out = fopen("sriram_gpu", "wb");
		fwrite(compressedDataOffset, sizeof(unsigned int),inputFileLength + 1, out);
		fclose(out);
	}	

	//index of each compressed block
	unsigned int *arrayCompressedBlocksLength = (unsigned int *)malloc((numInputDataBlocks) * sizeof(unsigned int));
	for(unsigned int i = 0; i < numInputDataBlocks; i++){
		if(inputBlocksIndex[i + 1] > inputBlocksIndex[i]){
			arrayCompressedBlocksLength[i] = (inputBlocksIndex[i + 1] - inputBlocksIndex[i]) / 8;
		}
		else{
			arrayCompressedBlocksLength[i] = inputBlocksIndex[i + 1] / 8;
		}
	}

	//launch cuda
	if(numKernelRuns == 1 && numIntegerOverflows == 0){
		cuda_compress_single_run_no_overflow(inputFileLength, numInputDataBlocks, inputFileData, compressedDataOffset, huffmanDictionary, frequency, argv[2], arrayCompressedBlocksLength);
	}
	else if(numKernelRuns == 1 && numIntegerOverflows == 1){
		cuda_compress_single_run_with_overflow(inputFileLength, numInputDataBlocks, inputFileData, compressedDataOffset, huffmanDictionary, frequency, argv[2], arrayCompressedBlocksLength, integerOverFlowIndex[0]);	
	}
	else if(numKernelRuns != 1 && numIntegerOverflows == 0){
		cuda_compress_multiple_run_no_overflow(inputFileLength, numInputDataBlocks, inputFileData, compressedDataOffset, huffmanDictionary, frequency, argv[2], arrayCompressedBlocksLength, numKernelRuns, gpuOverFlowIndex);
	}
	else if(numKernelRuns != 1 && numIntegerOverflows != 0){
		cuda_compress_multiple_run_with_overflow(inputFileLength, numInputDataBlocks, inputFileData, compressedDataOffset, huffmanDictionary, frequency, argv[2], arrayCompressedBlocksLength, numKernelRuns, gpuOverFlowIndex, numIntegerOverflows, integerOverFlowIndex);
	}

	// calculate run duration
	end = clock();

	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedDataOffset);
	return 0;
}
