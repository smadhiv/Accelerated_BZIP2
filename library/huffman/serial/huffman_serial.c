#include "huffman_serial.h"

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
				struct huffmanTree temp_huffmanTreeNode = huffmanTreeNode[b];
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
void build_huffman_dictionary(huffmanTree_t *root, unsigned char *bitSequence, unsigned char bitSequenceLength, huffmanDictionary_t *huffmanDictionary){
	if (root->left){
		bitSequence[bitSequenceLength] = 0;
		build_huffman_dictionary(root->left, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->right){
		bitSequence[bitSequenceLength] = 1;
		build_huffman_dictionary(root->right, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->left == NULL && root->right == NULL){
		huffmanDictionary[root->letter].bitSequenceLength = bitSequenceLength;
		memcpy(huffmanDictionary[root->letter].bitSequence, bitSequence, bitSequenceLength * sizeof(unsigned char));
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//builds the output data 
unsigned int generate_compressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *compressedBlockData, huffmanDictionary_t *huffmanDictionary){
	unsigned char writeBit = 0, bitsFilled = 0;
	unsigned int compressedBlockLength = 0;

	for (unsigned int i = 0; i < inputBlockLength; i++){
		for (unsigned int j = 0; j < huffmanDictionary[inputBlockData[i]].bitSequenceLength; j++){
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

	if(distinctCharacterCount == 1){
		head_huffmanTreeNode = &huffmanTreeNode[0];
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
