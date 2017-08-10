/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//Header used for serial and MPI implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#ifndef MIN_SCRATCH_SIZE
#define MIN_SCRATCH_SIZE 50 * 1024 * 1024
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 900000
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 1024
#endif

#ifndef GRID_DIM
#define GRID_DIM 4
#endif
//dictionary struct that holds the sequence and its length
typedef struct {
	unsigned char bitSequence[256][191];
	unsigned char bitSequenceLength[256];
} huffmanDictionary_t;

//huffmantree node struct that  holds the character and its frequency
typedef struct huffmanTree
{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
} huffmanTree_t;
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//helper functions
void intitialize_frequency(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData);
unsigned int intitialize_huffman_tree_get_distinct_char_count(unsigned int *frequency, huffmanTree_t *huffmanTreeNode);
void sort_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int combinedHuffmanNodes, huffmanTree_t *huffmanTreeNode);
void build_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int combinedHuffmanNodes, huffmanTree_t *huffmanTreeNode, huffmanTree_t **head_huffmanTreeNode);
void build_huffman_dictionary(huffmanTree_t *root, unsigned char *bitSequence, unsigned char bitSequenceLength, huffmanDictionary_t *huffmanDictionary);
unsigned int generate_compressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *compressedBlockData, huffmanDictionary_t *huffmanDictionary);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* compressedBlockData);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int compute_mem_offset(unsigned int *frequency, huffmanDictionary_t *huffmanDictionary);
void create_data_offset_array(int index, unsigned int *compressedDataOffset, unsigned char* inputBlockData, unsigned int inputBlockLength, huffmanDictionary_t *huffmanDictionary, unsigned int *integerOverFlowBlockIndex, unsigned int *numIntegerOverflows, unsigned int *kernelOverFlowIndex, unsigned int *numKernelRuns, long unsigned int *mem_used, long unsigned int mem_avail);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
int build_compressed_data_offset(unsigned int *compressedDataOffset, unsigned int *inputBlocksIndex, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, unsigned char *inputBlockData, unsigned int inputFileLength, unsigned char *inputFileData, unsigned int *numIntegerOverflows,  unsigned int *integerOverFlowIndex, unsigned int *numKernelRuns,  unsigned int *kernelOverFlowIndex, long unsigned int mem_avail);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void cuda_compress_single_run_no_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength);
void cuda_compress_single_run_with_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int integerOverFlowIndex);
void cuda_compress_multiple_run_no_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int numKernelRuns, unsigned int *gpuOverFlowIndex);
void cuda_compress_multiple_run_with_overflow(unsigned int inputFileLength, unsigned int numInputDataBlocks, unsigned char *inputFileData, unsigned int *compressedDataOffset, huffmanDictionary_t *huffmanDictionary, unsigned int **frequency, char *outputFileName, unsigned int *arrayCompressedBlocksLength, unsigned int numKernelRuns, unsigned int *gpuOverFlowIndex, unsigned int numIntegerOverflows, unsigned int *integerOverFlowIndex);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void encode_single_run_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks);
__global__ void compress_single_run_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int inputFileLength);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void encode_single_run_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int numInputDataBlocks, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow);
__global__ void compress_single_run_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void	encode_multiple_runs_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputBlockLength, unsigned int numInputDataBlocks, unsigned int lowerBlock, unsigned int upperBlock, unsigned int d_inputFileLength);
__global__ void compress_multiple_runs_no_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int d_inputBlockLength, unsigned int d_writePosition);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void	encode_multiple_runs_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, huffmanDictionary_t *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputBlockLength, unsigned int numInputDataBlocks, unsigned int lowerBlock, unsigned int upperBlock, unsigned int overFlowBlock, unsigned char *d_byteCompressedData_overflow, unsigned int d_inputFileLength);
__global__ void	compress_multiple_runs_with_overflow(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, unsigned char *d_byteCompressedData, unsigned int d_inputBlockLength, unsigned int d_writePosition, unsigned int overFlowBlock, unsigned int upperBlock, unsigned char *d_byteCompressedData_overflow);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/