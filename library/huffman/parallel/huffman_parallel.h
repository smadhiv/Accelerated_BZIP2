/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//Header used for cuda implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//dictionary struct that holds the sequence and its length
struct huffmanDictionary
{
	unsigned char bitSequence[256][191];
	unsigned char bitSequenceLength[256];
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
unsigned int generate_uncompressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *outputBlockData, struct huffmanTree *head_huffmanTreeNode);
int wrapperGPU(char **file, unsigned char *inputFileData, int inputFileLength);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* compressedBlockData);
void reverse_huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* outputBlockData);
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

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
// get bitSequence sequence for each character value
void build_huffman_dictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength, 	struct huffmanDictionary *huffmanDictionary){
	if (root->left){
		bitSequence[bitSequenceLength] = 0;
		build_huffman_dictionary(root->left, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->right){
		bitSequence[bitSequenceLength] = 1;
		build_huffman_dictionary(root->right, bitSequence, bitSequenceLength + 1, huffmanDictionary);
	}

	if (root->left == NULL && root->right == NULL){
		huffmanDictionary.bitSequenceLength[root->letter] = bitSequenceLength;
		if(bitSequenceLength < 192){
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
		}
		else{
			memcpy(bitSequenceConstMemory[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, 191);
			constMemoryFlag = 1;
		}
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//builds the output data 
unsigned int generate_compressed_data(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *compressedBlockData, struct huffmanDictionary *huffmanDictionary){
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
// the function calls above functions to generate compressed data
//returns the size of compressed data
unsigned int huffman_encoding(unsigned int *frequency, unsigned int inputBlockLength, unsigned char* inputBlockData, unsigned char* compressedBlockData){
	intitialize_frequency(frequency, inputBlockLength, inputBlockData);

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
	struct huffmanDictionary huffmanDictionary[256];
	unsigned char bitSequence[255], bitSequenceLength = 0;
	build_huffman_dictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength, 	huffmanDictionary);

	// compress
	//unsigned int compressedBlockLength = generate_compressed_data(inputBlockLength, inputBlockData, compressedBlockData, huffmanDictionary);
	//return compressedBlockLength;
	return 0;
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

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - single run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength){
	int i;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}		
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - single run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, int numBytes){
	int i, j;
	// calculate compressed data offset - (1048576 is a safe number that will ensure there is no integer overflow in GPU, it should be minimum 8 * number of threads)
	j = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
			integerOverflowIndex[j] = i;
			if(compressedDataOffset[i] % 8 != 0){
				bitPaddingFlag[j] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - multiple run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, long unsigned int mem_req){
	int i, j;
	j = 0;
	gpuMemoryOverflowIndex[0] = 0;
	gpuBitPaddingFlag[0] = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] > mem_req){
			gpuMemoryOverflowIndex[j * 2 + 1] = i;
			gpuMemoryOverflowIndex[j * 2 + 2] = i + 1;
			if(compressedDataOffset[i] % 8 != 0){
				gpuBitPaddingFlag[j + 1] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
	gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - multiple run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, int numBytes, long unsigned int mem_req){
	int i, j, k;
	j = 0;
	k = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(j != 0 && ((long unsigned int)compressedDataOffset[i + 1] + compressedDataOffset[integerOverflowIndex[j - 1]] > mem_req)){
			gpuMemoryOverflowIndex[k * 2 + 1] = i;
			gpuMemoryOverflowIndex[k * 2 + 2] = i + 1;
			if(compressedDataOffset[i] % 8 != 0){
				gpuBitPaddingFlag[k + 1] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			k++;
		}
		else if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
			integerOverflowIndex[j] = i;
			if(compressedDataOffset[i] % 8 != 0){
				bitPaddingFlag[j] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];	
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
	gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*---------------------------------------------------------------------------------------------------------------------------------------------*/