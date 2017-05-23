/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//reverse BWT Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 900000

unsigned char buffer[BLOCK_SIZE + 1];
unsigned int T[BLOCK_SIZE + 1];
unsigned int buffer_length;
unsigned int frequency[257];
unsigned int RunningTotal[257];

int main(int argc, char ** argv){
  FILE *input_file, *output_file;
  unsigned int first, last;
  unsigned int i, j;

  //check for arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open input file
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

//
  while( fread(&buffer_length, sizeof(unsigned int), 1, input_file) ){
    fread(buffer, sizeof(char), buffer_length, input_file);
    fread(&first, sizeof(unsigned int), 1, input_file);
    fread(&last, sizeof(unsigned int), 1, input_file);

    //initialize frequency with zero
    for(i = 0; i < 257; i++){
      frequency[i] = 0;
    }

    //get frequency for each symbol
    for(i = 0; i < buffer_length; i++){
      if(i == last){
        frequency[256]++;
      }
      else{
        frequency[buffer[i]]++;
      }
    }

    //get running total
    unsigned int sum = 0;
    for(i = 0; i < 257; i++){
      RunningTotal[i] = sum;
      sum += frequency[i];
      frequency[i] = 0;
    }

    //get the transformation vector
    for(i = 0; i < buffer_length; i++){
      if(i == last){
      }
      else{
        T[frequency[buffer[i]] + RunningTotal[buffer[i]]] = i;
        frequency[buffer[i]]++;
      }
    }

    //get the output
    i = first;
    for ( j = 0 ; j < buffer_length - 1; j++ ) {
      putc( buffer[ i ], output_file);
      i = T[ i ];
    }
  }
  fclose(input_file);
  fclose(output_file);
  return 0;
}