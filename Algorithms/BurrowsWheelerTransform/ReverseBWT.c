/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//reverse BWT Implementation
//convert putc to fwrite
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900000
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../Headers/ubwt.h"

int main(int argc, char ** argv){
  //files for i/o
  FILE *input_file, *output_file;
  //time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;
  //first and last characters  are written to output to aid BWT
  unsigned char inputBlockData[BLOCK_SIZE + 9];
  unsigned char outputBlockData[BLOCK_SIZE];
  unsigned int inputBlockLength;

  //check for arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open i/o files
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

	// start time measure
	start = clock();
  
  //reverse BWT in loop for each block in input file
  while( (inputBlockLength = fread(inputBlockData, sizeof(unsigned char), BLOCK_SIZE + 9, input_file)) ){
    reverse_burrows_wheeler_transform(inputBlockLength, inputBlockData, outputBlockData);
    fwrite(outputBlockData, sizeof(unsigned char), inputBlockLength - 9, output_file);
  }

	// end time measure
	end = clock();

  //close i/o files
  fclose(input_file);
  fclose(output_file);
  
  //compute and print run time
	cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);
  return 0;
}
