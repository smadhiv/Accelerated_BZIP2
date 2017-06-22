/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//BWT Implementation
//convert putc to fwrite
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 900000
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Headers/bwt.h"

//do BWT
int main(int argc, char **argv){
  //files for i/o
  FILE *input_file, *output_file;
	//time measurement
	clock_t start, end;
	unsigned int cpuTimeUsed;
  //store output data
  unsigned char outputBlockData[BLOCK_SIZE + 9];
  
  //check for required arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open i/o file
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

	// start time measure
	start = clock();
  
  //perform BWT in the while loop
  //read BLOCK_SIZE at a time, stop when there is nothing to read
  while( (inputBlockLength = fread(inputBlockData, 1, BLOCK_SIZE, input_file)) ){
    burrows_wheeler_transform(outputBlockData);
    fwrite(outputBlockData, sizeof(unsigned char), inputBlockLength + 9, output_file);
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

