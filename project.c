#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "mpi.h"

#define FILENAME_SIZE 256
#define READ_BUF 1024
#define MASTER 0

typedef struct {
    char filename[FILENAME_SIZE];
    off_t size_in_bytes;
} FileInfo;

typedef struct {
    off_t start_offset;
    off_t end_offset;
    int numfiles;
} ProcessIndex;


int main(int argc, char **argv) {
    // All processes
    int rank, numtasks, tag = 1;
    off_t batch_size, total_size, remainder;
    MPI_Datatype process_data, file_info;
    MPI_Aint extent, lb;
    int blocklengths[3];
    MPI_Datatype types[3];
    MPI_Aint displacements[3];
    ProcessIndex pindex;

    // Master only
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

    // ProcessIndex struct type
    MPI_Type_get_extent(MPI_INT64_T, &lb, &extent);
    
    blocklengths[0] = 1;
    types[0] = MPI_INT64_T;
    displacements[0] = 0;

    blocklengths[1] = 1;
    types[1] = MPI_INT64_T;
    displacements[1] = blocklengths[0] * extent;

    blocklengths[2] = 1;
    types[2] = MPI_INT;
    displacements[2] = displacements[1] + blocklengths[1] * extent;

    MPI_Type_create_struct(3, blocklengths, displacements, types, &process_data);
    MPI_Type_commit(&process_data);

    // FileInfo struct
    MPI_Type_get_extent(MPI_CHAR, &lb, &extent);
    blocklengths[0] = FILENAME_SIZE;
    types[0] = MPI_CHAR;
    displacements[0] = 0;

    blocklengths[1] = 1;
    types[1] = MPI_INT64_T;
    displacements[1] = blocklengths[0] * extent;

    MPI_Type_create_struct(2, blocklengths, displacements, types, &file_info);
    MPI_Type_commit(&file_info);

    // Comm info
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

        // FileInfo reading
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
            
            files[size - 1].filename[0] = '\0';
            strcat(files[size - 1].filename, in_file->d_name);
            files[size - 1].size_in_bytes = buf.st_size;
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
        off_t nextfile_size = files[0].size_in_bytes;
        // Files schedulation
        for(int i = 0, j = 0; i < numtasks; i++) {
            process_indexes[i].start_offset = next;  // Process i begin offset
            displs[i] = j;
            offsets[i] = 0;

            int mybatch = batch_size + ((remainder > i));   // Bytes of process i
            
            while(j < size && mybatch > 0) {
                offsets[i] += 1;                // Assegna il file j al processore i
                
                if(mybatch <= nextfile_size) {       
                    process_indexes[i].end_offset = files[j].size_in_bytes - nextfile_size + mybatch; // Calcolo la posizione in cui sono arrivato nell'ultimo file 
                    next = (mybatch == nextfile_size) ? 0 : process_indexes[i].end_offset + 1;       // Offset di partenza del prossimo processo
                    nextfile_size -= mybatch;

                    // We don't need to procede to next file
                    if(nextfile_size != 0) {
                        break;
                    }
                    mybatch = 0;    // To exit from while
                }
                
                mybatch -= nextfile_size;
                j += 1;
                if(j < size)
                    nextfile_size = files[j].size_in_bytes;

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
            printf("%ld-%ld-%d ", process_indexes[i].start_offset, process_indexes[i].end_offset, process_indexes[i].numfiles);
        }
        printf("\n");
        
    }

    // Processes indexes 
    MPI_Scatter(process_indexes, 1, process_data, &pindex, 1, process_data, MASTER, MPI_COMM_WORLD);
    
    FileInfo *recvfiles = malloc(sizeof(FileInfo) * pindex.numfiles);
    // printf("Task %d : %ld-%ld-%d\n", rank, pindex.start_offset, pindex.end_offset, pindex.numfiles);
    
    // Files info
    MPI_Scatterv(files, offsets, displs, file_info, recvfiles, pindex.numfiles, file_info, MASTER, MPI_COMM_WORLD);
    
    for(int i = 0; i < pindex.numfiles; i++) {
        printf("Task %d: %s (%ld)\n", rank, recvfiles[i].filename, recvfiles[i].size_in_bytes);
    }

    // Computation
    FILE *fp;
    char filename[FILENAME_SIZE];
    char readbuf[READ_BUF];
    int offset = pindex.start_offset;
    int rd;

    for(int i = 0; i < pindex.numfiles; i++) {
        filename[0] = '\0';
        strcat(filename, argv[1]);
        strcat(filename,"/");
        strcat(filename, recvfiles[i].filename);

        if((fp = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: fopen error on file %s\n", filename);
            
            return EXIT_FAILURE;
        }
        
        if(fseek(fp, offset, SEEK_SET) != 0) {
            fprintf(stderr, "Error: fseek error on file %s\n", filename);
            
            return EXIT_FAILURE;
        }

        while(rd = fgets(readbuf, READ_BUF, fp)) {
            //TODO: Continuare e selezionare le parole e vedere come usare un dizionario da inviare poi
            printf("task %d: \n%s\n", rank, readbuf);
        }

        fclose(fp);
        offset = 0;
    }

    MPI_Type_free(&process_data);
    MPI_Type_free(&file_info);
    MPI_Finalize();

    if(rank == MASTER) {
        free(files);
        free(displs);
        free(offsets);
        free(process_indexes);
    }

    return EXIT_SUCCESS;
}