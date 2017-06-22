/*******************************************************************************************/
//global variables to be used in qsort function
unsigned int inputBlockLength;
//store input block data
unsigned char inputBlockData[BLOCK_SIZE];
/*******************************************************************************************/

/*******************************************************************************************/
//BWT functions
void burrows_wheeler_transform(unsigned char *outputBlockData);
//for the qsort
int compare( const unsigned int *index1, const unsigned int *index2 );
//to store int in char array
void serialize_int(unsigned char *outputBlockData, unsigned int value, unsigned int *count);
/*******************************************************************************************/


/*******************************************************************************************/
//BWT
void burrows_wheeler_transform(unsigned char *outputBlockData){
  //first and last characters  are written to output to aid BWT
  //store indices that correspond to each character in input data
  unsigned int first;
  unsigned int last;
  unsigned int indices[BLOCK_SIZE + 1];

  //initialize indices. Since we are going to sort the input block in increasing order, 
  //the indeces are initialized with values (0 - buffer length - 1), these numbers 
  //correspond to the input characters
  //remember we need the last character and not the first
  for(unsigned int i = 0; i < inputBlockLength + 1; i++){
    indices[i] = i;
  }

  //sort the buffer and store the result as indices in the array
  qsort(indices, inputBlockLength + 1, sizeof(int), ( int (*)(const void *, const void *) )compare);

  //first hold the positon where the first character is 
  //last stores the index of the end of input buffer
  //write each character to file 
  unsigned int count = 0;
  for (unsigned int i = 0 ; i <= inputBlockLength ; i++){
    if(indices[ i ] != 0){
      outputBlockData[count++] = inputBlockData[ indices[ i ] - 1 ];
      if(indices[i] == 1){
        //capture first, the first element is present at the end of the string that starts with the index 1
        first = i;
      }
    }
    else{
      //capture last, the last element is present at the end of the string that starts with the zero index
      last = i;
      outputBlockData[count++] = '?';
    }
  }
  //printf("first = %u\n", first);
  //printf("last = %u\n", last);
  serialize_int(outputBlockData, first, &count);
  serialize_int(outputBlockData, last, &count);
}
/*******************************************************************************************/

/*******************************************************************************************/
//for the qsort, we do memcmp to compare each mem location, if all same, then which ever comes earlier is the lesser
//we do not wrap around the string to make compare
int compare(const unsigned int *index1, const unsigned int *index2){
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
/*******************************************************************************************/

/*******************************************************************************************/
//write int into char array
void serialize_int(unsigned char *outputBlockData, unsigned int value, unsigned int *count){
//Write big-endian int value into buffer; assumes 32-bit int and 8-bit char.
  outputBlockData[(*count)++] = value;
  outputBlockData[(*count)++] = value >> 8;
  outputBlockData[(*count)++] = value >> 16;
  outputBlockData[(*count)++] = value >> 24;
}
/*******************************************************************************************/