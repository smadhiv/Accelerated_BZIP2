/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//BWT Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 900000

long length;
unsigned char buffer[BLOCK_SIZE];
unsigned int indices[BLOCK_SIZE + 1];

//for the qsort
int bounded_compare( const unsigned int *i1, const unsigned int *i2 );

//do BWT
int main(int argc, char **argv){
  int i;
  FILE *input_file, *output_file;
  
  //check for arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open files file
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

  //perform bwt in the while loop
  //read BLOCK_SIZE at a time, stop when there is nothing to read
  while( length = fread(buffer, 1, BLOCK_SIZE, input_file) ){
    //initialize indices
    for(i = 0; i < length + 1; i++){
      indices[i] = i;
    }

    //sort the buffer and store the result as indices in the array
    qsort(indices, length + 1, sizeof(int), ( int (*)(const void *, const void *) )bounded_compare);

    //write to output the length
    unsigned int temp = length + 1;
    fwrite( &temp, 1, sizeof( unsigned int ), output_file );

    //first hold the positon where the first character is 
    //last stores the index of the end of input buffer
    unsigned int first;
    unsigned int last;
    for (i = 0 ; i <= length ; i++ ) {
      if ( indices[ i ] == 1 ){
        first = i;
      }

      if ( indices[ i ] == 0 ) {
        last = i;
        fputc('?', output_file);
      } 
      else{
        fputc(buffer[ indices[ i ] - 1 ], output_file);
      }
    }
    fwrite( &first, 1, sizeof( unsigned int ), output_file );   
    fwrite( &last, 1, sizeof( unsigned int ), output_file );
  }

  return 0;
}

//for the qsort
int bounded_compare( const unsigned int *i1, const unsigned int *i2 ){
  unsigned int l1 = (unsigned int) ( length - *i1 );
  unsigned int l2 = (unsigned int) ( length - *i2 );

  int result = memcmp( buffer + *i1, buffer + *i2, l1 < l2 ? l1 : l2 );
  if ( result == 0 ){
    return l2 - l1;
  }
  else{
    return result;
  }
}
