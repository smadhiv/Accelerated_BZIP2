/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU kernels
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#ifndef huffman_parallel
#include "../../../library/huffman/parallel/huffman_parallel.h"
#endif

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run and no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//encode
__global__ void encode_single_run_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks){
	__shared__ huffmanDictionary_t d_huffmanDictionary_shared;
	unsigned int inputFileLength = d_inputFileLength;
	unsigned int pos = threadIdx.x;

	for(unsigned int i = blockIdx.x; i < numInputDataBlocks; i += gridDim.x){  
		//copy the specific dictionary to the shared memory
		if(threadIdx.x == 0){
    	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));
  	}
		__syncthreads();
    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;
		//copy the input char's encoded bytes into d_byteCompressedData
	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
		__syncthreads();
  }


	pos = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int upperLimit = d_compressedDataOffset[inputFileLength];
 	for(unsigned int i = pos * 8; i < upperLimit; i += (blockDim.x * gridDim.x) * 8){
	  for(unsigned int j = 0; j < 8; j++){
		  if(d_byteCompressedData[i + j] == 0){
			  d_inputFileData[i / 8] = d_inputFileData[i / 8] << 1;
		  }
		  else{
			  d_inputFileData[i / 8] = (d_inputFileData[i / 8] << 1) | 1;
		  }
	  }
  }
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run and no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//compress
/*
__global__ void compress_single_run_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int inputFileLength){
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int upperLimit = d_compressedDataOffset[inputFileLength];
 for(unsigned int i = pos * 8; i < upperLimit; i += (blockDim.x * gridDim.x) * 8){
	  for(unsigned int j = 0; j < 8; j++){
		  if(d_byteCompressedData[i + j] == 0){
			  d_inputFileData[i / 8] = d_inputFileData[i / 8] << 1;
		  }
		  else{
			  d_inputFileData[i / 8] = (d_inputFileData[i / 8] << 1) | 1;
		  }
	  }
  }
}
*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//encode
__global__ void encode_single_run_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow){
	__shared__ huffmanDictionary_t d_huffmanDictionary_shared;
	unsigned int inputFileLength = d_inputFileLength;
	unsigned int pos = threadIdx.x;
  
	//till overflow
	for(unsigned int i = blockIdx.x; i < overFlowBlock; i += gridDim.x){
		//copy the specific dictionary to the shared memory
		if(threadIdx.x == 0){
    	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));
  	}
		__syncthreads();
    unsigned int upperLimit = i * BLOCK_SIZE + BLOCK_SIZE;
	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
  }

	//beyond overflow
	for(unsigned int i = blockIdx.x + overFlowBlock; i < numInputDataBlocks; i += gridDim.x){
		//copy the specific dictionary to the shared memory
		if(threadIdx.x == 0){
    	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));
  	}
		__syncthreads();
    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;
	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
				if(i == overFlowBlock && j == (i * BLOCK_SIZE)){
		  		for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  		d_byteCompressedData_overflow[k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  		}				
					continue;
				}
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData_overflow[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
  }
	__syncthreads();
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//compress
__global__ void compress_single_run_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow){
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int upperLimit_1 = d_compressedDataOffset[overFlowBlock * BLOCK_SIZE];
	for(unsigned int i = pos * 8; i < upperLimit_1; i += (blockDim.x * gridDim.x) * 8){
		for(unsigned int j = 0; j < 8; j++){
			if(d_byteCompressedData[i + j] == 0){
				d_inputFileData[i / 8] = d_inputFileData[i / 8] << 1;
			}
			else{
				d_inputFileData[i / 8] = (d_inputFileData[i / 8] << 1) | 1;
			}
		}
	}
	
	unsigned int offset_overflow = d_compressedDataOffset[overFlowBlock * BLOCK_SIZE] / 8;
	unsigned int upperLimit_2 = d_compressedDataOffset[d_inputFileLength];
	for(unsigned int i = pos * 8; i < upperLimit_2; i += (blockDim.x * gridDim.x) * 8){
		for(unsigned int j = 0; j < 8; j++){
			if(d_byteCompressedData_overflow[i + j] == 0){
				d_inputFileData[(i / 8) + offset_overflow] = d_inputFileData[(i / 8) + offset_overflow] << 1;
			}
			else{
				d_inputFileData[(i / 8) + offset_overflow] = (d_inputFileData[(i / 8) + offset_overflow] << 1) | 1;
			}
		}
	}
}
