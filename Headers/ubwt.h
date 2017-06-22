/***********************************************************************************************************/
void reverse_burrows_wheeler_transform(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *outputBlockData);
/***********************************************************************************************************/

/***********************************************************************************************************/
//perform reverse BWT
void reverse_burrows_wheeler_transform(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *outputBlockData){
  unsigned int inputDataFrequency[257];
  unsigned int runningTotal[257];
  unsigned int transformationVector[900001];
  //get position of first and last
  unsigned int* temp = (unsigned int* )&inputBlockData[inputBlockLength - 4];
  unsigned int last = *temp;
  unsigned int first = *(temp - 1);
  //printf("first = %u\n", first);
  //printf("last = %u\n", last);


  //initialize frequency with zeros
  //memset(inputDataFrequency, 0, 257 * sizeof(unsigned int));
  for(unsigned int i = 0; i < 257; i++){
     inputDataFrequency[i] = 0;
   }

  //get frequency for each symbol in the input block
  for(unsigned int i = 0; i < inputBlockLength - 8; i++){
    //we ignore the end of file indicator symbol at postion last
    if(i == last){
      inputDataFrequency[256]++;
    }
    else{
      inputDataFrequency[inputBlockData[i]]++;
    }
  }

  //get running total
  unsigned int sum = 0;
  for(unsigned int i = 0; i < 257; i++){
    runningTotal[i] = sum;
    sum += inputDataFrequency[i];
    inputDataFrequency[i] = 0;
  }

  //get the transformation vector
  //For a given row i, transformation vector[ i ] is defined as the row where string[ i + 1 ] is found
  for(unsigned int i = 0; i < inputBlockLength - 8; i++){
    if(i == last){
    }
    else{
      transformationVector[inputDataFrequency[inputBlockData[i]] + runningTotal[inputBlockData[i]]] = i;
      inputDataFrequency[inputBlockData[i]]++;
    }
  }

  //get the output
  unsigned int count = 0;
  unsigned int i = first;
  for (unsigned int j = 0 ; j < inputBlockLength - 9; j++) {
    outputBlockData[count++] = inputBlockData[i];
    i = transformationVector[i];
  }
}
/***********************************************************************************************************/