#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "mpi.h"

#define FILENAME_SIZE 256
#define MASTER 0

typedef struct {
    char filename[FILENAME_SIZE];
    off_t sizeInBytes;
} FileInfo;

typedef struct {
    off_t startOffset;
    off_t endOffset;
    int numfiles;
} ProcessIndex;


int main(int argc, char **argv) {
    int rank, numtasks, tag = 1;
    off_t batch_size, total_size, remainder;
    // Master
    DIR* FD;
    struct dirent* in_file;
    struct stat buf;
    FileInfo *files;
    ProcessIndex *process_indexes;
    int *offsets, *displs;


    if(argc != 2) {
        fprintf(stderr, "Error : pass a directory path\n");

        return EXIT_FAILURE;
    }

    
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == MASTER) {
        if((FD = opendir(argv[1])) == NULL) {
            fprintf(stderr, "Error : Failed to open input directory\n");

            return EXIT_FAILURE;
        }

        files = malloc(sizeof(FileInfo));
        int size = 0;
        total_size = 0;

        while((in_file = readdir(FD))) {
            char file[256] = {'\0'};

            if (!strcmp (in_file->d_name, "."))
                continue;
            if (!strcmp (in_file->d_name, ".."))    
                continue;

            size += 1;
            files = realloc(files, sizeof(FileInfo) * size);
        
            strcat(file, argv[1]);
            strcat(file, "/");

            if(stat(strcat(file, in_file->d_name), &buf) != 0)
                fprintf(stderr, "Error: Stat error on file %s\n", in_file->d_name);

            strcat(files[size - 1].filename, in_file->d_name);
            files[size - 1].sizeInBytes = buf.st_size;
            total_size += buf.st_size;
            
            printf("%s - %ld\n", in_file->d_name, buf.st_size);
        }

        printf("File read: %d\n", size);
        printf("Size in bytes: %ld\n", total_size);

        batch_size = total_size / numtasks;
        remainder = total_size % batch_size;

        printf("batch_size: %ld\n", batch_size);

        offsets = malloc(sizeof(int) * numtasks);
        displs = malloc(sizeof(int) * numtasks);
        
        process_indexes = malloc(sizeof(ProcessIndex) * numtasks);

        off_t next = 0;
        off_t nextfile_size = files[0].sizeInBytes;
        for(int i = 0, j = 0; i < numtasks; i++) {
            process_indexes[i].startOffset = next;
            displs[i] = j;
            offsets[i] = 0;

            int mybatch = batch_size + ((remainder > i));
            
            while(j < size && mybatch > 0) {
                offsets[i] += 1;                // Assegna il file j al processore i
                
                if(mybatch <= nextfile_size) {       //TODO: Non modificare la size dei file
                    process_indexes[i].endOffset = files[j].sizeInBytes - nextfile_size + mybatch; // Calcolo la posizione in cui sono arrivato nell'ultimo file 
                    next = (mybatch == nextfile_size) ? 0 : process_indexes[i].endOffset + 1;
                    nextfile_size -= mybatch;
                    
                    if(nextfile_size == 0) {
                        j += 1;
                        nextfile_size = files[j].sizeInBytes;
                    
                    }
                    break;
                }
                
                mybatch -= nextfile_size;
                j += 1;
                if(j < size)
                    nextfile_size = files[j].sizeInBytes;
            }

            process_indexes[i].numfiles = offsets[i];
        }

        printf("Displs: ");
        for(int i = 0; i < numtasks; i++) {
            printf("%d ", displs[i]);
        }
        printf("\nOffsets: ");
        for(int i = 0; i < numtasks; i++) {
            printf("%d ", offsets[i]);
        }
        printf("\nProcess indexes:");
        for(int i = 0; i < numtasks; i++) {
            printf("%ld-%ld-%d ", process_indexes[i].startOffset, process_indexes[i].endOffset, process_indexes[i].numfiles);
        }
        printf("\n");
        
    }



    MPI_Finalize();

    if(rank == MASTER) {
        free(files);
        free(displs);
    }

    return EXIT_SUCCESS;
}