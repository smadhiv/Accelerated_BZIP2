#include "uhuffman_serial.h"

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//intitialize huffmantree nodes with the character and its frequency
//returns the number of distinct values in the given input data
unsigned int intitialize_huffman_tree_get_distinct_char_count(unsigned int *frequency, struct huffmanTree *huffmanTreeNode){
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
void sort_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int mergedHuffmanNodes, struct huffmanTree *huffmanTreeNode){
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
void build_huffman_tree(unsigned int i, unsigned int distinctCharacterCount, unsigned int mergedHuffmanNodes, struct huffmanTree *huffmanTreeNode, struct huffmanTree **head_huffmanTreeNode){
	huffmanTreeNode[distinctCharacterCount + i].count = huffmanTreeNode[mergedHuffmanNodes].count + huffmanTreeNode[mergedHuffmanNodes + 1].count;
	huffmanTreeNode[distinctCharacterCount + i].left = &huffmanTreeNode[mergedHuffmanNodes];
	huffmanTreeNode[distinctCharacterCount + i].right = &huffmanTreeNode[mergedHuffmanNodes + 1];
	*head_huffmanTreeNode = &(huffmanTreeNode[distinctCharacterCount + i]);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//builds the uncompressed data
unsigned int generate_uncompressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *outputBlockData, struct huffmanTree *head_huffmanTreeNode){
	struct huffmanTree *current_huffmanTreeNode = head_huffmanTreeNode;
	unsigned int outputBlockLength = 0;
	for (unsigned int i = 0; i < inputBlockLength; i++){
		unsigned char currentInputByte = inputBlockData[i];
		for (unsigned int j = 0; j < 8; j++){
			unsigned char currentInputBit = currentInputByte & 0200;
			currentInputByte = currentInputByte << 1;
			if (currentInputBit == 0){
				current_huffmanTreeNode = current_huffmanTreeNode->left;
				if (current_huffmanTreeNode->left == NULL){
					outputBlockData[outputBlockLength++] = current_huffmanTreeNode->letter;
					current_huffmanTreeNode = head_huffmanTreeNode;
				}
			}
			else{
				current_huffmanTreeNode = current_huffmanTreeNode->right;
				if (current_huffmanTreeNode->right == NULL){
					outputBlockData[outputBlockLength++] = current_huffmanTreeNode->letter;
					current_huffmanTreeNode = head_huffmanTreeNode;
				}
			}
		}
	}
	return outputBlockLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// the function calls above functions to generate uncompressed data
//returns the size of uncompressed data
void reverse_huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* outputBlockData){
	struct huffmanTree huffmanTreeNode[512];
	unsigned int distinctCharacterCount = intitialize_huffman_tree_get_distinct_char_count(frequency, huffmanTreeNode);

	// build tree 
	struct huffmanTree *head_huffmanTreeNode = NULL;
	for (unsigned int i = 0; i < distinctCharacterCount - 1; i++){
		unsigned int combinedHuffmanNodes = 2 * i;
		sort_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode);
		build_huffman_tree(i, distinctCharacterCount, combinedHuffmanNodes, huffmanTreeNode, &head_huffmanTreeNode);
	}

	// write the data to file
	generate_uncompressed_data(inputBlockLength, inputBlockData, outputBlockData, head_huffmanTreeNode);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/