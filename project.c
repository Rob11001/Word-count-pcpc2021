#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <ctype.h>

#include "mpi.h"
#include "./libs/uthash.h"

#define FILENAME_SIZE 256
#define READ_BUF 1024
#define WORD_SIZE 256
#define MASTER 0

// Structures
typedef struct {
    char filename[FILENAME_SIZE];
    off_t size_in_bytes;
} FileInfo;

typedef struct {
    off_t start_offset;
    off_t end_offset;
    int numfiles;
} ProcessIndex;

typedef struct {
    char word[WORD_SIZE];           /* key */
    int counts;
    UT_hash_handle hh;  /* makes this structure hashable */ 
} MapEntry;

typedef struct {
    char word[WORD_SIZE];
    int counts;
} Couple;

// Utility functions
void add_word(MapEntry **map, char* word, int counts);
void increase_word_counter(MapEntry **map, char *word, int counts);
int num_of_bytes_UTF8(char first_char);
int computeAndMap(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path);
int gatheringAndReduce(MapEntry **master_map, int master, MapEntry **local_map, int rank, int numtasks, int count_tag, int send_tag, MPI_Datatype couple_type, MPI_Datatype couple_type_resized);
int receiveAndreduce(MapEntry **map, int size, int source, int tag, MPI_Datatype type);
int map_cmp(MapEntry *a, MapEntry *b);

int main(int argc, char **argv) {
    // All processes
    int rank, numtasks, send_tag = 1, count_tag = 2, rc;
    off_t batch_size, total_size, remainder;
    MPI_Datatype process_data, file_info, couple_type, couple_type_resized;
    MPI_Aint extent, lb;
    int blocklengths[3];
    MPI_Datatype types[3];
    MPI_Aint displacements[3];
    ProcessIndex pindex;
    MapEntry *local_map = NULL;

    // Master only
    DIR* FD;
    struct dirent* in_file;
    struct stat buf;
    FileInfo *files;
    ProcessIndex *pindexes;
    int *offsets, *displs;
    MapEntry *master_map = NULL;

    if(argc != 2) {
        fprintf(stderr, "Error : pass a directory path\n");

        return EXIT_FAILURE;
    }

    // MPI init
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

    // Couple struct
    blocklengths[0] = WORD_SIZE;
    types[0] = MPI_CHAR;
    displacements[0] = 0;

    blocklengths[1] = 1;
    types[1] = MPI_INT;
    displacements[1] = blocklengths[0] * extent;
    
    MPI_Type_create_struct(2, blocklengths, displacements, types, &couple_type);
    MPI_Type_commit(&couple_type);    
    MPI_Type_get_extent(couple_type, &lb, &extent);
    MPI_Type_create_resized(couple_type, lb, sizeof(MapEntry), &couple_type_resized);
    MPI_Type_commit(&couple_type_resized);    
    
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
        
        pindexes = malloc(sizeof(ProcessIndex) * numtasks);

        off_t next = 0;
        off_t nextfile_size = files[0].size_in_bytes;
        // Files schedulation
        for(int i = 0, j = 0; i < numtasks; i++) {
            pindexes[i].start_offset = next;  // Process i begin offset
            displs[i] = j;
            offsets[i] = 0;

            int mybatch = batch_size + ((remainder > i));   // Bytes of process i
            
            while(j < size && mybatch > 0) {
                offsets[i] += 1;                // Assegna il file j al processore i
                
                if(mybatch <= nextfile_size) {       
                    pindexes[i].end_offset = files[j].size_in_bytes - nextfile_size + mybatch; // Calcolo la posizione in cui sono arrivato nell'ultimo file 
                    next = (mybatch == nextfile_size) ? 0 : pindexes[i].end_offset + 1;       // Offset di partenza del prossimo processo
                    nextfile_size -= mybatch;

                    // We don't need to procede to next file
                    if(nextfile_size != 0) {
                        break;
                    }
                    mybatch = 0;    // To exit from the cycle
                }
                
                mybatch -= nextfile_size;
                j += 1; // Pass to next file
                if(j < size)
                    nextfile_size = files[j].size_in_bytes; //Next file's size

            }

            pindexes[i].numfiles = offsets[i];
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
            printf("%ld-%ld-%d ", pindexes[i].start_offset, pindexes[i].end_offset, pindexes[i].numfiles);
        }
        printf("\n");
        
    }

    // Processes indexes 
    if((rc = MPI_Scatter(pindexes, 1, process_data, &pindex, 1, process_data, MASTER, MPI_COMM_WORLD)) != MPI_SUCCESS)
        return rc;
    
    FileInfo *recvfiles = malloc(sizeof(FileInfo) * pindex.numfiles);
    
    // Files info
    if((rc = MPI_Scatterv(files, offsets, displs, file_info, recvfiles, pindex.numfiles, file_info, MASTER, MPI_COMM_WORLD))!= MPI_SUCCESS)
        return rc;

    // Computation and word mapping
    if((rc = computeAndMap(recvfiles, pindex.numfiles, pindex.start_offset, pindex.end_offset, &local_map, argv[1])) != 0) {
        fprintf(stderr, "Computation error, error code: %d\n", rc);

        return EXIT_FAILURE;
    }
    
    printf("Task %d -> total_word: %d\n", rank, HASH_COUNT(local_map));

    //TODO: generazione csv file da parte del master
    //TODO: riordinare e sistemare il codice in maniera più pulita e leggibile
    //TODO: Vedere cos'altro fare


    // Gathering and Reduce (TODO: Aggiustare nome funzione e parametri)
    if((rc = gatheringAndReduce(&master_map, MASTER, &local_map, rank, numtasks, count_tag, send_tag, couple_type, couple_type_resized)) != MPI_SUCCESS)
        return rc;

    if(rank == MASTER) {
        HASH_SORT(master_map, map_cmp);
        printf("Num: %d\n", HASH_COUNT(master_map));
        for(MapEntry *p = master_map; p != NULL; p = p->hh.next) 
            printf("%s %d\n", p->word, p->counts);
    }

    fflush(stdout);

    MPI_Type_free(&process_data);
    MPI_Type_free(&file_info);
    MPI_Type_free(&couple_type);
    MPI_Type_free(&couple_type_resized);
    MPI_Finalize();

    if(rank == MASTER) {
        free(files);
        free(displs);
        free(offsets);
        free(pindexes);

        // Free hash
        for(MapEntry *e = master_map, *next; e != NULL; e = next) {
            next = e->hh.next;
            free(e);
        }
    }

    // Free hash
    for(MapEntry *e = local_map, *next; e != NULL; e = next) {
        next = e->hh.next;
        free(e);
    }
    
    return EXIT_SUCCESS;
}


// Functions

void add_word(MapEntry **map, char* word_str, int counts) {
    MapEntry *s = malloc(sizeof(MapEntry));
    
    strcpy(s->word, word_str);
    s->counts = counts;

    HASH_ADD_STR(*map, word, s);
}

void increase_word_counter(MapEntry **map, char *word, int counts) {
    MapEntry *entry = NULL;
    
    HASH_FIND_STR(*map, word, entry);
    if(entry != NULL) 
        entry->counts += counts;
     else 
        add_word(map, word, counts);
}

int num_of_bytes_UTF8(char first_char) {
    int byte = first_char & 0b11110000;
    switch(byte) {
        case 192: // Two bytes
            return 2;
        case 224: // Three bytes
            return 3;
        case 240: // Four bytes
            return 4;
        default:  // Not used
            return 1;
    }
}

int computeAndMap(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path) {
    FILE *fp;
    char filename[FILENAME_SIZE];
    char readbuf[READ_BUF] = {'\0'};
    long offset = start_offset, rd = 0;

    for(int i = 0; i < num_files; i++) {
        // Filename construction
        filename[0] = '\0';
        strcat(filename, dir_path);
        strcat(filename,"/");
        strcat(filename, files[i].filename);

        // File open
        if((fp = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: fopen error on file %s\n", filename);
            
            return EXIT_FAILURE;
        }

        if(fseek(fp, offset > 0 ? offset - 1 : offset, SEEK_SET) != 0) {
            fprintf(stderr, "Error: fseek error on file %s\n", filename);
            
            return EXIT_FAILURE;
        }

        char currentWord[WORD_SIZE] = {'\0'};
        int currentWordSize = 0, jump = 0, done = 0;
       
        // File reading
        while(fgets(readbuf, READ_BUF, fp) && !jump) {  // jump boolean variable to skip the cycle
            char *p = readbuf;
            if(!done && i == 0 && offset > 0) {  // To handle word conflicts between processes
                rd -= 1;
                
                while(isalpha(*p) || *p < 0) {// oppure è un carattere UTF-8 con più bytes
                    p += 1;
                    rd += 1;
                }
                
                if(rd < 0) {    // To set the correct rd and p value
                    rd = 0;
                    p += 1;
                }

                done = 1;
            }
            
            // Reading from the buffer
            for(; *p && p < readbuf + READ_BUF; p++) {
                rd += 1;
                
                if(isalpha(*p)) {   // One byte char
                    currentWord[currentWordSize++] = tolower(*p); // Legge carattere per carattere
                } else if(*p < 0) { // Multi byte char
                    int len = num_of_bytes_UTF8(*p);
                    for(int j = 0; j < len; j++)
                        currentWord[currentWordSize++] = tolower(*(p + j));
                    p += (len - 1);
                } else if(currentWordSize > 0) { // Word ended
                    // Ho una parola
                    currentWord[currentWordSize] = '\0';
                    increase_word_counter(map, currentWord, 1);
                    currentWord[0] = '\0';
                    currentWordSize = 0;
                    
                    if(i  == num_files - 1 && (rd + offset > end_offset)) {   // To handle last file end_offset(also to handle processes conflicts)
                        jump = 1;   // boolean flag to skip the outer cycle
                        break;
                    }
                }
            }
            readbuf[0] = '\0';  // Buffer reset
        }

        // If I end to read the file and there is still a word in currentWord buffer
        if(currentWordSize > 0) {
            currentWord[currentWordSize] = '\0';
            increase_word_counter(map, currentWord, 1);
            currentWord[0] = '\0';
            currentWordSize = 0;
        }

        fclose(fp);
        rd = 0;
        offset = 0;
    }

    return 0;
}

int gatheringAndReduce(MapEntry **master_map, int master, MapEntry **local_map, int rank, int numtasks, int count_tag, int send_tag, MPI_Datatype couple_type, MPI_Datatype couple_type_resized) {
    int rc = 0;
    
    if(rank == master) {
        // Request and counts
        MPI_Request *reqs = malloc(sizeof(MPI_Request) * (numtasks - 1));
        int *counts = malloc(sizeof(int) * (numtasks - 1));

        // Post for counts
        for(int p = 0, i = 0; p < numtasks; p++) {
            if(p == master)
                continue;
            
            if((rc = MPI_Irecv(counts + i, 1, MPI_INT, p, count_tag, MPI_COMM_WORLD, reqs + i)) != MPI_SUCCESS)
                return rc;
            i += 1;
        }

        int received = 0;
        // Reduces the own local_map (Not really necessary, we could re-use local_map)
        for(MapEntry *e = *local_map; e != NULL; e = e->hh.next) {
            int index, flag;

            increase_word_counter(master_map, e->word, e->counts);

            if((rc = MPI_Testany(numtasks - 1, reqs, &index, &flag, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
                return rc;
            
            if(index != MPI_UNDEFINED) {
                if((rc = receiveAndreduce(master_map, counts[index], index + 1, send_tag, couple_type)) != MPI_SUCCESS)
                    return rc;

                received += 1;
            } 

        }

        // Receives and reduces all the remains data
        while(received != numtasks - 1) {
            int flag, index;
            
            // Waits until an receive has been completed
            if((rc = MPI_Waitany(numtasks - 1, reqs, &index, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
                return rc;
            // Gains and reduces the data
            if((rc = receiveAndreduce(master_map, counts[index], index + 1, send_tag, couple_type)) != MPI_SUCCESS)
                return rc;

            received += 1; // Updates counter
        }
        
        free(reqs);
        free(counts);

    } else {
        int size = HASH_COUNT(*local_map), i = 0;

        // Flattening of own Hashmap
        MapEntry *list_to_send = malloc(sizeof(MapEntry) * size);
        for(MapEntry *e = *local_map; e != NULL; e = e->hh.next)
            list_to_send[i++] = *e;
        
        // Sends the size of its map
        if((rc = MPI_Send(&size, 1, MPI_INT, MASTER, count_tag, MPI_COMM_WORLD)) != MPI_SUCCESS)
            return rc;
        // Sends the map using a resized datatype to skip one parameter of the struct
        if((rc = MPI_Send(list_to_send, size, couple_type_resized, MASTER, send_tag, MPI_COMM_WORLD)) != MPI_SUCCESS)
            return rc;

        free(list_to_send);
    }

    return rc;
}

int receiveAndreduce(MapEntry **map, int size, int source, int tag, MPI_Datatype type) {
    int rc = 0;
    Couple *buf = malloc(sizeof(Couple) * size);
            
    if((rc = MPI_Recv(buf, size, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
        return rc;
    
    for(int i = 0; i < size; i++)
        increase_word_counter(map, buf[i].word, buf[i].counts);  
    free(buf);

    return rc;
}

int map_cmp(MapEntry *a, MapEntry *b) {
    return (b->counts - a->counts);
}