/******************************************************************************************/
//Sriram Madhivanan
//RLE compression
//header has file length information
//marker is 0
//for example 111234 -> 3(1)0(234)0
//012340033 -> 0(001234)02(0)2(3)
/******************************************************************************************/
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
  //declarations
  unsigned int fileLength, compressedFileLength;
  unsigned char runLength = 1, characterEncoded;
  unsigned char *inputFileData, *compressedFileData;
  FILE *inputFile, *compressedFile;
	unsigned char marker = 0;
	int flag = 0;
	
  //verify input args
  if(argc != 3){
	  printf("\nProvide input and output file names");
	  return -1;
  }
  
  //file read 
  inputFile = fopen(argv[1], "rb");
  compressedFile = fopen(argv[2], "wb");
  if (inputFile == NULL || compressedFile == NULL){
    printf("\nUnable to open file\n");
    return -1;
  }
  
  //get file fileLength
  fseek(inputFile, 0, SEEK_END);
  fileLength = ftell(inputFile);
  
  //allocate memrory for file
  inputFileData = malloc(fileLength * sizeof(unsigned char));
  compressedFileData = malloc(fileLength * 2 * sizeof(unsigned char));
  
  //read input data
  fseek(inputFile, 0, SEEK_SET);
  fread(inputFileData, sizeof(unsigned char), fileLength, inputFile);

  //compress data
  compressedFileLength = 0;
  characterEncoded = inputFileData[0];
				
	for(unsigned int i = 1; i < fileLength; i++){
		if(flag == 0){
			if(inputFileData[i] == characterEncoded){
				runLength++;
			}
			else{
				if(runLength == 1){
					flag = 1;
					compressedFileData[compressedFileLength++] = marker;
					compressedFileData[compressedFileLength++] = characterEncoded;
					if(characterEncoded == marker){
						compressedFileData[compressedFileLength++] = marker;
					}
					characterEncoded = inputFileData[i];
				}
				else{
					compressedFileData[compressedFileLength++] = runLength;
					compressedFileData[compressedFileLength++] = characterEncoded;
					runLength = 1;
					characterEncoded = inputFileData[i];
				}
			}
		}
		else{
			if(inputFileData[i] == characterEncoded){
				runLength++;
				compressedFileData[compressedFileLength++] = marker;
				flag = 0;
			}
			else{
				compressedFileData[compressedFileLength++] = characterEncoded;
				if(characterEncoded == marker){
					compressedFileData[compressedFileLength++] = marker;		
				}
				characterEncoded = inputFileData[i];
			}
		}
	}
	
  //the last byte
	if (flag == 0){
		compressedFileData[compressedFileLength++] = runLength;
		compressedFileData[compressedFileLength++] = characterEncoded;		
	}
	else{
		compressedFileData[compressedFileLength++] = characterEncoded;
		compressedFileData[compressedFileLength++] = marker;
	}

	
  //write the compressedFileData file
	fwrite(&fileLength, sizeof(unsigned int), 1, compressedFile);
  fwrite(compressedFileData, sizeof(unsigned char), compressedFileLength, compressedFile);
	free(inputFileData);
	free(compressedFileData);
  fclose(compressedFile);
  fclose(inputFile);
	return 0;
}