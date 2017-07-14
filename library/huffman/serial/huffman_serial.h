/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//Header used for serial and MPI implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 900000
#endif

//dictionary struct that holds the sequence and its length
struct huffmanDictionary
{
	unsigned char bitSequence[255];
	unsigned char bitSequenceLength;
};

//huffmantree node struct that  holds the character and its frequency
struct huffmanTree
{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
};
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//helper functions
void intitialize_frequency(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData);
unsigned int intitialize_huffman_tree_get_distinct_char_count(unsigned int *frequency, struct huffmanTree *huffmanTreeNode);
void sort_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int combinedHuffmanNodes, struct huffmanTree *huffmanTreeNode);
void build_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int combinedHuffmanNodes, struct huffmanTree *huffmanTreeNode, struct huffmanTree **head_huffmanTreeNode);
void build_huffman_dictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength, 	struct huffmanDictionary *huffmanDictionary);
unsigned int generate_compressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *compressedBlockData, struct huffmanDictionary *huffmanDictionary);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* compressedBlockData);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
