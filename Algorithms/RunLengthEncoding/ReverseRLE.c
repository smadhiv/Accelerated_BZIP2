/******************************************************************************************/
//Sriram Madhivanan
//RLE decompression
/******************************************************************************************/
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
  //declarations
  unsigned int fileLength, decompressedFileLength;
  //unsigned char runLength, characterEncoded;
  unsigned char *inputFileData, *decompressedFileData;
  FILE *inputFile, *decompressedFile;

  //verify input args
  if(argc != 3){
	  printf("\nProvide input and output file names");
	  return -1;
  }
  
  //file read 
  inputFile = fopen(argv[1], "rb");
  decompressedFile = fopen(argv[2], "wb");
  if (inputFile == NULL || decompressedFile == NULL){
    printf("\nUnable to open file\n");
    return -1;
  }
  
  //get file fileLength
  fseek(inputFile, 0, SEEK_END);
  fileLength = ftell(inputFile);
  fseek(inputFile, 0, SEEK_SET);
  
  //read decompressed file length
  fread(&decompressedFileLength, sizeof(unsigned int), 1, inputFile);

  //allocate memrory for file
  inputFileData = malloc(fileLength * sizeof(unsigned char));
  fread(inputFileData, sizeof(unsigned char), fileLength, inputFile);
  decompressedFileData = malloc(decompressedFileLength * sizeof(unsigned char));

  //decompress data
	int flag = 0;
	int pos = 0;
	unsigned char marker = 0;
  for(unsigned int i = 0; i < fileLength; i++){
		if(flag == 0){
			if(inputFileData[i] == marker){
				flag = 1;
			}
			else{
				for(unsigned int j = 0; j < inputFileData[i]; j++){
					decompressedFileData[pos++] = inputFileData[i + 1];
				}
				i++;
			}
		}
		else{
			if(inputFileData[i] != marker){
				decompressedFileData[pos++] = inputFileData[i];
			}
			else if(inputFileData[i] == marker && i < fileLength - 1 && inputFileData[i + 1] == marker){
				decompressedFileData[pos++] = marker;
				i++;
			}
			else{
				flag = 0;
			}
		}
	}
	
	//write the decompressedFileData file
  fwrite(decompressedFileData, sizeof(unsigned char) * decompressedFileLength, 1, decompressedFile);
	free(inputFileData);
	free(decompressedFileData);
  fclose(inputFile);
  fclose(decompressedFile);
	return 0;
}