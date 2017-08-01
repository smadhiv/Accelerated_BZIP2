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
__global__ void compress_single_run_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks){
	__shared__ huffmanDictionary_t d_huffmanDictionary_shared;

	unsigned int inputFileLength = d_inputFileLength;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  
	for(unsigned int i = blockIdx.x; i < numInputDataBlocks; i += blockDim.x){
    
		//copy the specific dictionary to the shared memory
  	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));

    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
  }

	__syncthreads();
 for(unsigned int i = pos * 8; i < d_compressedDataOffset[inputFileLength]; i += blockDim.x * 8){
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
// single run with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void compress_single_run_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow){
	__shared__ huffmanDictionary_t d_huffmanDictionary_shared;

	unsigned int inputFileLength = d_inputFileLength;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  
	for(unsigned int i = blockIdx.x; i < overFlowBlock; i += blockDim.x){
    
		//copy the specific dictionary to the shared memory
  	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));

    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
  }

	for(unsigned int i = blockIdx.x + overFlowBlock; i < numInputDataBlocks; i += blockDim.x){
    
		//copy the specific dictionary to the shared memory
  	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));

    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData_overflow[d_compressedDataOffset[j] + k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
		  }
	  }
  }

	__syncthreads();
	
	for(unsigned int i = pos * 8; i < d_compressedDataOffset[overFlowBlock * BLOCK_SIZE]; i += blockDim.x * 8){
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
	
	for(unsigned int i = pos * 8; i < d_compressedDataOffset[inputFileLength]; i += blockDim.x * 8){
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
