/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//reverse BWT Implementation
//convert putc to fwrite
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 900000

unsigned char inputBlockData[BLOCK_SIZE + 1];
unsigned int T[BLOCK_SIZE + 1];
unsigned int inputBlockLength;
unsigned int frequency[257];
unsigned int RunningTotal[257];

int main(int argc, char ** argv){
  //files for i/o
  FILE *input_file, *output_file;
  //first and last characters  are written to output to aid BWT
  unsigned int first;
  unsigned int last;

  //check for arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open i/o files
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

  //reverse BWT in loop for each block in input file
  while( fread(&inputBlockLength, sizeof(unsigned int), 1, input_file) ){
    fread(inputBlockData, sizeof(unsigned char), inputBlockLength, input_file);
    fread(&first, sizeof(unsigned int), 1, input_file);
    fread(&last, sizeof(unsigned int), 1, input_file);

    //initialize frequency with zeros
    for(unsigned int i = 0; i < 257; i++){
      frequency[i] = 0;
    }

    //get frequency for each symbol in the input block
    for(unsigned int i = 0; i < inputBlockLength; i++){
      //we ignore the end of file indicator symbol at postion last
      if(i == last){
        frequency[256]++;
      }
      else{
        frequency[inputBlockData[i]]++;
      }
    }

    //get running total
    unsigned int sum = 0;
    for(unsigned int i = 0; i < 257; i++){
      RunningTotal[i] = sum;
      sum += frequency[i];
      frequency[i] = 0;
    }

    //get the transformation vector
    //For a given row i, transformation vector[ i ] is defined as the row where string[ i + 1 ] is found
    for(unsigned int i = 0; i < inputBlockLength; i++){
      if(i == last){
      }
      else{
        T[frequency[inputBlockData[i]] + RunningTotal[inputBlockData[i]]] = i;
        frequency[inputBlockData[i]]++;
      }
    }

    //get the output
    unsigned int i = first;
    for (unsigned int j = 0 ; j < inputBlockLength - 1; j++) {
      putc( inputBlockData[i], output_file);
      i = T[i];
    }
  }

  //close i/o files
  fclose(input_file);
  fclose(output_file);
  return 0;
}