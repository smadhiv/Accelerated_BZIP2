/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MTF Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// linked list to store dictionary
struct dictionaryLinkedList{
	unsigned char val;
	struct dictionaryLinkedList* next;
	struct dictionaryLinkedList* prev;
};

// search and return the index
unsigned char searchValue(struct dictionaryLinkedList* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(dictionary->val != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return index;
}

// swap dictionary and returns head
struct dictionaryLinkedList* swapValue(struct dictionaryLinkedList* dictionary, unsigned char indexToSwap){
	unsigned int index = 0;
	struct dictionaryLinkedList* head = dictionary;
	while(index != indexToSwap){
		dictionary = dictionary->next;
		index++;
	}

	if(dictionary != head){
		// take care of previous and next node to the current
		dictionary->prev->next = dictionary->next;
		if(dictionary->next != NULL){
			dictionary->next->prev = dictionary->prev;
		}

		// take care of current
		dictionary->prev = NULL;
		dictionary->next = head;
		head->prev = dictionary;
	}
	return dictionary;
}

// check linked list data
void check_linked_list(struct dictionaryLinkedList* head){
	struct dictionaryLinkedList* current;
	printf("checking data\n");
	current = head;
	while(current->next != NULL){
		printf("%u\n", current->val);
		current = current->next;
	}
	printf("%u\n", current->val);
}

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	int dictionary[256];
	unsigned int uniqueChars = 0, inputFileLength;
	unsigned char *inputFileData, *outputDataIndex;
	FILE *inputFile, *outFile;
	struct dictionaryLinkedList *head, *current, *tail;

	// check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	// read input file, get filelength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = malloc(inputFileLength * sizeof(unsigned char));
	outputDataIndex = malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);	

	// start time measure
	start = clock();
	
	// initialize dictionary
	for (unsigned int i = 0; i < 256; i++){
	    dictionary[i] = -1;
	}

	// generate dictionary of each symbols
	for (unsigned int i = 0; i < inputFileLength; i++){
		if(dictionary[inputFileData[i]] == -1){
			dictionary[inputFileData[i]] = inputFileData[i];
			uniqueChars++;
		}
	}

	// initialize dictionary
	for (unsigned int i = 0; i < 256; i++){
		if(dictionary[i] == -1){
			unsigned int j = i + 1;
			while(dictionary[j] == -1 && j < 256){
				j++;
			}
			if(j < 256){
				dictionary[i] = dictionary[j];
				dictionary[j] = -1;
			}
			else{
				break;
			}
		}
	}

	for(unsigned int i = 0; i < uniqueChars; i++){
		printf("%u\n", dictionary[i]);
	}

	// store dictionary into linked list
	current = malloc(sizeof(struct dictionaryLinkedList));
	current->val = dictionary[0];
	current->prev = NULL;
	head = current;

	for(unsigned int i = 1; i < uniqueChars; i++){
		struct dictionaryLinkedList *node = malloc(sizeof(struct dictionaryLinkedList));
		node->val = dictionary[i];
		current->next = node;
		node->prev = current;
		current = node;
	}
	current->next = NULL;
	tail = current;

	// generate array list of indeces
	for(unsigned int i = 0; i < inputFileLength; i++){
		outputDataIndex[i] = searchValue(head, inputFileData[i]);
		head = swapValue(head, outputDataIndex[i]);
	}

	// write to output
	outFile = fopen(argv[2], "wb");
	fwrite(&uniqueChars, sizeof(unsigned int), 1, outFile);
	fwrite(dictionary, sizeof(int), uniqueChars, outFile);
	fwrite(outputDataIndex, sizeof(unsigned char), inputFileLength, outFile);
	
	// end time measure
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	
	fclose(outFile);
	free(inputFileData);
	free(outputDataIndex);
    return 0;
}