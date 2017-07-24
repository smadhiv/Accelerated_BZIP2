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
__global__ void compress(unsigned int *d_inputBlocksIndex, unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks){
	__shared__ huffmanDictionary_t d_huffmanDictionary_shared;

	unsigned int inputFileLength = d_inputFileLength;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  
	for(unsigned int i = blockIdx.x; i < numInputDataBlocks; i += blockDim.x){
    
		//copy the specific dictionary to the shared memory
  	memcpy(&d_huffmanDictionary_shared, &d_huffmanDictionary[i], sizeof(huffmanDictionary_t));

    unsigned int upperLimit = i < numInputDataBlocks - 1 ? i * BLOCK_SIZE + BLOCK_SIZE : inputFileLength;

	  for(unsigned int j = (i * BLOCK_SIZE) + pos; j < upperLimit; j += blockDim.x){
		  for(unsigned int k = 0; k < d_huffmanDictionary_shared.bitSequenceLength[d_inputFileData[j]]; k++){
			  d_byteCompressedData[d_compressedDataOffset[j]+k] = d_huffmanDictionary_shared.bitSequence[d_inputFileData[j]][k];
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

