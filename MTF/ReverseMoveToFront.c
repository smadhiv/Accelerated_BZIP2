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
unsigned char searchIndex(struct dictionaryLinkedList* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(index != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return dictionary->val;
}

// swap dictionary and returns head
struct dictionaryLinkedList* swapIndex(struct dictionaryLinkedList* dictionary, unsigned char value){
	struct dictionaryLinkedList* head = dictionary;
	while(value != dictionary->val){
		dictionary = dictionary->next;
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
	unsigned int uniqueChars, inputFileLength, outputFileLength;
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
	
    fread(&uniqueChars, sizeof(unsigned int), 1, inputFile);
    fread(&dictionary, sizeof(int), uniqueChars, inputFile);

	for(unsigned int i = 0; i < uniqueChars; i++){
		printf("%u\n", dictionary[i]);
	}

    outputFileLength = inputFileLength - uniqueChars*sizeof(int) - sizeof(uniqueChars);
    inputFileData = malloc(outputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), outputFileLength, inputFile);
	fclose(inputFile);

	outputDataIndex = malloc(outputFileLength * sizeof(unsigned char));
	// start time measure
	start = clock();
	
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
	for(unsigned int i = 0; i < outputFileLength; i++){
		outputDataIndex[i] = searchIndex(head, inputFileData[i]);
		head = swapIndex(head, outputDataIndex[i]);
	}

	// write to output
	outFile = fopen(argv[2], "wb");
	fwrite(outputDataIndex, sizeof(unsigned char), outputFileLength, outFile);
	
	// end time measure
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	
	fclose(outFile);
	free(inputFileData);
	free(outputDataIndex);
    return 0;
}
