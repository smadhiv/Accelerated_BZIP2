/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//BWT Implementation
//convert putc to fwrite
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 900000
//the size  of block read (<= BLOCK_SIZE)
unsigned int inputBlockLength;
//store input block data
unsigned char inputBlockData[BLOCK_SIZE];
//store indices that correspond to each character in input data
unsigned int indices[BLOCK_SIZE + 1];

//for the qsort
int compar( const unsigned int *index1, const unsigned int *index2 );

//do BWT
int main(int argc, char **argv){
  //files for i/o
  FILE *input_file, *output_file;
  //first and last characters  are written to output to aid BWT
  unsigned int first;
  unsigned int last;

  //check for required arguments
  if(argc != 3){
    printf("Error: Need input and output file names\n");
    return -1;
  }

  //open i/o file
  input_file = fopen(argv[1], "rb");
  output_file = fopen(argv[2], "wb");

  //perform BWT in the while loop
  //read BLOCK_SIZE at a time, stop when there is nothing to read
  while( (inputBlockLength = fread(inputBlockData, 1, BLOCK_SIZE, input_file)) ){
    //initialize indices. Since we are going to sort the input block in increasing order, 
    //the indeces are initialized with values (0 - buffer length - 1), these numbers 
    //correspond to the input characters
    //remember we need the last character and not the first
    for(unsigned int i = 0; i < inputBlockLength + 1; i++){
      indices[i] = i;
    }

    //sort the buffer and store the result as indices in the array
    qsort(indices, inputBlockLength + 1, sizeof(int), ( int (*)(const void *, const void *) )compar);

    //write to output the length
    unsigned int temp = inputBlockLength + 1;
    fwrite(&temp, 1, sizeof( unsigned int ), output_file);

    //first hold the positon where the first character is 
    //last stores the index of the end of input buffer
    //write each character to file 
    for (unsigned int i = 0 ; i <= inputBlockLength ; i++){
      if(indices[ i ] != 0){
        fputc(inputBlockData[ indices[ i ] - 1 ], output_file);
        if(indices[i] == 1){
          //capture first, the first element is present at the end of the string that starts with the index 1
          first = i;
        }
      }
      else{
        //capture last, the last element is present at the end of the string that starts with the zero index
        last = i;
        fputc('?', output_file);
      }
    }
    fwrite(&first, 1, sizeof(unsigned int), output_file);   
    fwrite(&last, 1, sizeof(unsigned int), output_file);
  }

  //close i/o files
  fclose(input_file);
  fclose(output_file);
  return 0;
}

//for the qsort, we do memcmp to compare each mem location, if all same, then which ever comes earlier is the lesser
//we do not wrap around the string to make compare
int compar(const unsigned int *index1, const unsigned int *index2){
  unsigned int length1 = (unsigned int) (inputBlockLength - *index1);
  unsigned int length2 = (unsigned int) (inputBlockLength - *index2);

  int result = memcmp(inputBlockData + *index1, inputBlockData + *index2, length1 < length2 ? length1 : length2);
  if (result == 0){
    return length2 - length1;
  }
  else{
    return result;
  }
}
