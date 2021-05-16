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
#define WORD_SIZE 100
#define MASTER 0

// Structures

/* contains the required info for describing a file*/
typedef struct {
    char filename[FILENAME_SIZE];
    off_t size_in_bytes;
} FileInfo;

/* The info required from a process to know its section of data */
typedef struct {
    off_t start_offset;
    off_t end_offset;
    int numfiles;
} ProcessIndex;

/* A entry of the Hash. It's a couple key-value. The key is a string, the value an integer */
typedef struct {
    char word[WORD_SIZE];           /* key */
    int counts;
    UT_hash_handle hh;  /* makes this structure hashable */ 
} MapEntry;

/* A struct which represents a simple couple string-int */
typedef struct {
    char word[WORD_SIZE];
    int counts;
} Couple;

// Utility functions

/**
 * @brief Splits up the files between the numtasks processors
 * 
 * @param numtasks the number of processors and size od the offsets, displs and pindexes arrays
 * @param send_counts an array of size 'numtasks', which will contain the number of file assigned to processors
 * @param displs an array of size 'numtasks', which will contain the displacements of processors respect to files
 * @param pindexes an array of size 'numtasks', which will contain the required info for every process
 * @param total_size the total size of files in bytes used to compute batch size. Every processor will be given at least total_size/numtasks bytes
 * @param filenum the size of the files array 
 * @param files an array of size 'filenum' 
 */
void file_scheduling(int numtasks, int *send_counts, int *displs, ProcessIndex *pindexes, int total_size, int filenum, FileInfo *files);

/**
 * @brief Adds the key to the hash with the passed value
 * 
 * @param map the hash
 * @param word the key
 * @param counts the value 
 */
void add_word(MapEntry **map, char* word, int counts);

/**
 * @brief Increases the value of the entry with the passed key, or creates it if necessary
 * 
 * @param map the hash
 * @param word the key
 * @param counts the value to add to old value
 */
void increase_word_counter(MapEntry **map, char *word, int counts);

/**
 * @brief Calculate the number of bytes of the character using UTF-8
 * 
 * @param first_char char
 * @return int value between 1 and 4
 */
int num_of_bytes_UTF8(char first_char);

/**
 * @brief Checks if is a symbol
 * 
 * @param ch 
 * @return [0-1]
 */
int issymbol(char ch);

/**
 * @brief Checks is is a symbol
 * 
 * @param ch 
 * @return [0-1]
 */
int ismulticharsymbol(char *ch);

/**
 * @brief Reads all words in the section of data passed and puts them in the hash
 * 
 * @param files list of file to read
 * @param num_files the number of files
 * @param start_offset the starting offset of the fist file
 * @param end_offset the ending offset of the last file
 * @param map the hash
 * @param dir_path the dir path in which files are stored
 * @param rank (Used for debugging)
 * @return int 0 if is all okay, non-zero number otherwise
 */
int computeAndMap(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path, int rank);

/**
 * @brief All processes send their data to the master, then the master collects them and reduces them in
 * only one hash
 * 
 * @param master_map the hash used to collect all data (only used by the master)
 * @param master the rank of the master
 * @param recv_type datatype used to recv data (only used by the master)
 * @param local_map the map to send to the master
 * @param rank the rank of the process
 * @param send_type datatype used to send data 
 * @param numtasks the total number of processes in the comm
 * @param size_tag tag used for the messages which processes use to specify the size of the data to send
 * @param send_tag tag used to send the real data
 * @param comm the communicator used
 * @return int 0 if it's all okay, non-zero otherwise
 */
int gatheringAndReduce(MapEntry **master_map, int master, MPI_Datatype recv_type, MapEntry **local_map, int rank, MPI_Datatype send_type, int numtasks, int size_tag, int send_tag, MPI_Comm comm);

/**
 * @brief Utility function which handles the receive of a list of couple from a process
 * 
 * @param map the hash in which the couples will be added
 * @param size the number of couples to receive
 * @param source the process from which receives the data
 * @param tag the tag used in the communication
 * @param type the datatype used 
 * @param comm the communicator used
 * @return int 0 if it's all okay, non-zero otherwise
 */
int receiveMap(MapEntry **map, int size, int source, int tag, MPI_Datatype type, MPI_Comm comm);

/**
 * @brief Compares two Map Entry using counts 
 * 
 * @param a Map entry
 * @param b Map entry
 * @return int 0 if are equals, > 0 if a is less frequent than b, < 0 otherwise
 */
int map_cmp(MapEntry *a, MapEntry *b);

/**
 * @brief Create a csv file
 * 
 * @param filename name of the file
 * @param map the map will be written in the file
 * @return int 0 if it's all okay, non-zero otherwise
 */
int create_csv(char *filename, MapEntry *map);

int main(int argc, char **argv) {
    // All processes
    int rank, numtasks, send_tag = 1, count_tag = 2, rc;
    int degree = 1, *dest, master = MASTER;
    double start, end;
    off_t batch_size, total_size, remainder;
    MPI_Datatype process_data, file_info, couple_type, couple_type_resized;
    MPI_Aint extent, lb;
    int blocklengths[3];
    MPI_Datatype types[3];
    MPI_Aint displacements[3];
    ProcessIndex pindex;
    MapEntry *local_map = NULL;
    MPI_Group graph_group, world_group;
    MPI_Comm graph_comm;

    // Master only
    DIR* FD;
    struct dirent* in_file;
    struct stat buf;
    FileInfo *files;
    ProcessIndex *pindexes;
    int *send_counts, *displs;
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

    // Couple struct resized     
    MPI_Type_get_extent(couple_type, &lb, &extent);
    MPI_Type_create_resized(couple_type, lb, sizeof(MapEntry), &couple_type_resized);
    MPI_Type_commit(&couple_type_resized);    
    
    // Comm info
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Graph topology construction
    if(rank == MASTER) {
        degree = numtasks - 1;
        dest = malloc(sizeof(int) *degree);
        for(int i = 0, j = 0; i < numtasks; i++)
            if(i != rank)
                dest[j++] = i;
    } else {
        dest = malloc(sizeof(int));
        *dest = MASTER;
    }

    if((rc = MPI_Dist_graph_create(MPI_COMM_WORLD, 1, &rank, &degree, dest, MPI_UNWEIGHTED, MPI_INFO_NULL, 1, &graph_comm)) != MPI_SUCCESS)
        return rc;
    
    free(dest);

    // Ranks' updating
    MPI_Comm_group(graph_comm, &graph_group);
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_translate_ranks(world_group, 1, &master, graph_group, &master); 
    MPI_Comm_rank(graph_comm, &rank);  

    // Master reads input directory
    if(rank == master) {
        if((FD = opendir(argv[1])) == NULL) {
            fprintf(stderr, "Error : Failed to open input directory\n");

            return EXIT_FAILURE;
        }

        files = malloc(sizeof(FileInfo));
        int size = 0;
        total_size = 0;

        // FileInfo reading (directory reading)
        while((in_file = readdir(FD))) {
            char file[256] = {'\0'};

            if(in_file->d_type != DT_REG)
                continue;
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
            
            // For debugging purpose
            printf("%s - %ld\n", in_file->d_name, buf.st_size);
        }

        // For debugging purpose
        batch_size = total_size / numtasks;
        remainder = total_size % batch_size;
        printf("File read: %d\n", size);
        printf("Size in bytes: %ld\n", total_size);
        printf("batch_size: %ld\n", batch_size);

        // Files schedulation
        send_counts = malloc(sizeof(int) * numtasks);
        displs = malloc(sizeof(int) * numtasks);
        pindexes = malloc(sizeof(ProcessIndex) * numtasks);
        
        file_scheduling(numtasks, send_counts, displs, pindexes, total_size, size, files);

        // For debugging purpose
        printf("Displs: ");
        for(int i = 0; i < numtasks; i++) {
            printf("%d ", displs[i]);
        }
        printf("\nOffsets: ");
        for(int i = 0; i < numtasks; i++) {
            printf("%d ", send_counts[i]);
        }
        printf("\nProcess indexes:");
        for(int i = 0; i < numtasks; i++) {
            printf("%ld-%ld-%d ", pindexes[i].start_offset, pindexes[i].end_offset, pindexes[i].numfiles);
        }
        printf("\n");
        
    }

    printf("Execution task %d ...\n", rank);
    fflush(stdout);

    MPI_Barrier(graph_comm);
    start = MPI_Wtime();
    /* ************************************************* */

    // Processes indexes 
    if((rc = MPI_Scatter(pindexes, 1, process_data, &pindex, 1, process_data, master, graph_comm)) != MPI_SUCCESS)
        return rc;
    
    FileInfo *recvfiles = malloc(sizeof(FileInfo) * pindex.numfiles);
    
    // Files info
    if((rc = MPI_Scatterv(files, send_counts, displs, file_info, recvfiles, pindex.numfiles, file_info, master, graph_comm))!= MPI_SUCCESS)
        return rc;

    // Computation and word mapping
    if((rc = computeAndMap(recvfiles, pindex.numfiles, pindex.start_offset, pindex.end_offset, &local_map, argv[1], rank)) != 0) {
        fprintf(stderr, "Computation error, error code: %d\n", rc);

        return EXIT_FAILURE;
    }
    
    printf("Task %d -> total_word: %d\n", rank, HASH_COUNT(local_map));

    //TODO: Vedere cos'altro fare


    // Gathering and Reduce
    if((rc = gatheringAndReduce(&master_map, master, couple_type, &local_map, rank, couple_type_resized, numtasks, count_tag, send_tag, graph_comm)) != MPI_SUCCESS)
        return rc;

    if(rank == master) {
        HASH_SORT(master_map, map_cmp);
        int tot = 0;
        for(MapEntry *e = master_map; e != NULL; e = e->hh.next) {
            tot += e->counts;
        }
        printf("Unique-words: %d, Total words: %d\n", HASH_COUNT(master_map), tot);
        
        if((rc = create_csv(argv[1], master_map)) != 0) {
            fprintf(stderr, "Error in csv file creation\n");
            return rc;
        }
    }

    /* ************************************************* */
    MPI_Barrier(graph_comm);
    end = MPI_Wtime();
    fflush(stdout);

    MPI_Type_free(&process_data);
    MPI_Type_free(&file_info);
    MPI_Type_free(&couple_type);
    MPI_Type_free(&couple_type_resized);
    MPI_Comm_free(&graph_comm);
    MPI_Group_free(&graph_group);
    MPI_Group_free(&world_group);
    MPI_Finalize();

    if(rank == master) {
        // Time
        printf("Task %d -- Time in ms = %f\n", rank, end - start);
        
        // Deallocation
        free(files);
        free(displs);
        free(send_counts);
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

void file_scheduling(int numtasks, int *send_counts, int *displs, ProcessIndex *pindexes, int total_size, int numfiles, FileInfo *files) {
    int batch_size = total_size / numtasks;
    int remainder = total_size % batch_size;
    off_t next = 0;
    off_t nextfile_size = files[0].size_in_bytes;

    // Files schedulation
    for(int i = 0, j = 0; i < numtasks; i++) { // i - processes index, j - files index
        pindexes[i].start_offset = next;  // Process i begin offset
        displs[i] = j;
        send_counts[i] = 0;
        int mybatch = batch_size + ((remainder > i));   // Bytes of process i
        
        while(j < numfiles && mybatch > 0) {
            send_counts[i] += 1;                // Assegna il file j al processore i
            
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
            if(j < numfiles)
                nextfile_size = files[j].size_in_bytes; //Next file's size
        }
        pindexes[i].numfiles = send_counts[i];
    }
}

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


int issymbol(char ch) {
    return ch == '\'' || ch == '-';
}

int ismulticharsymbol(char *ch) {
    return strcmp(ch, "”") == 0 || strcmp(ch, "—") == 0 || strcmp(ch, "“") == 0;
}

int computeAndMap(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path, int rank) {
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

        char current_word[WORD_SIZE] = {'\0'};
        int current_word_size = 0, jump = 0, done = 0;
       
        // File reading
        while(fgets(readbuf, READ_BUF, fp) && !jump) {  // jump boolean variable to skip the cycle
            char *p = readbuf;
            if(!done && i == 0 && offset > 0) {  // To handle word conflicts between processes
                rd -= 1;
                
                while(isalpha(*p) || *p < 0 || issymbol(*p)) {// oppure è un carattere UTF-8 con più bytes oppure è un simbolo
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
                    current_word[current_word_size++] = tolower(*p); // Legge carattere per carattere
                } else if(*p < 0) { // Multi byte char
                    int len = num_of_bytes_UTF8(*p);
                    // checks current_word buffer overflow 
                    if(current_word_size + len < WORD_SIZE - 1) {
                        char ch[len + 1];
                        for(int j = 0; j < len; j++) {
                            ch[j] = *(p + j);
                            current_word[current_word_size++] = tolower(*(p + j));
                        }
                        ch[len] = '\0';

                        // If it's a symbol, it can be skipped
                        if(ismulticharsymbol(ch)) {
                            current_word_size -= len;
                        }

                        p += (len - 1);
                        rd += (len - 1); // Remember to update rd variable
                    } else {
                        p -= 1;
                        rd -= 1;
                        current_word[current_word_size] = '\0';
                        current_word_size = WORD_SIZE - 1;
                    }
                } else if(current_word_size > 0 && issymbol(*p) && !issymbol(current_word[current_word_size - 1])) {
                    current_word[current_word_size++] = *p;
                } else if(current_word_size > 0) { // Word ended
                    // Ho una parola
                    current_word[current_word_size] = '\0';
                    increase_word_counter(map, current_word, 1);
                    current_word[0] = '\0';
                    current_word_size = 0;    
                }

                // checks current_word buffer overflow 
                if(current_word_size == WORD_SIZE - 1) {
                        current_word[current_word_size] = '\0';
                        increase_word_counter(map, current_word, 1);
                        current_word[0] = '\0';
                        current_word_size = 0;
                }

                // If I don't have a word and I have gone beyond the end_offset I can stop
                if(i == (num_files - 1) && current_word_size == 0 && (rd + offset > end_offset)) {   // To handle last file end_offset(also to handle processes conflicts)
                    jump = 1;   // boolean flag to skip the outer cycle
                    break;
                }

            }
            readbuf[0] = '\0';  // Buffer reset
        }

        // If I end to read the file and there is still a word in currentWord buffer
        if(current_word_size > 0) {
            current_word[current_word_size] = '\0';
            increase_word_counter(map, current_word, 1);
            current_word[0] = '\0';
            current_word_size = 0;
        }

        fclose(fp);
        rd = 0;
        offset = 0;
    }

    return 0;
}

int gatheringAndReduce(MapEntry **master_map, int master, MPI_Datatype recv_type, MapEntry **local_map, int rank, MPI_Datatype send_type, int numtasks, int size_tag, int send_tag, MPI_Comm comm) {
    int rc = 0;
    
    if(rank == master) {
        // Request and counts
        MPI_Request *reqs = malloc(sizeof(MPI_Request) * (numtasks - 1));
        int *counts = malloc(sizeof(int) * (numtasks - 1));

        // Post for size
        for(int p = 0, i = 0; p < numtasks; p++) {
            if(p == master)
                continue;
            
            if((rc = MPI_Irecv(counts + i, 1, MPI_INT, p, size_tag, comm, reqs + i)) != MPI_SUCCESS)
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
                if((rc = receiveMap(master_map, counts[index], (index >= master) ? index + 1 : index, send_tag, recv_type, comm)) != MPI_SUCCESS)
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
            if((rc = receiveMap(master_map, counts[index], (index >= master) ? index + 1 : index, send_tag, recv_type, comm)) != MPI_SUCCESS)
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
        if((rc = MPI_Send(&size, 1, MPI_INT, master, size_tag, comm)) != MPI_SUCCESS)
            return rc;
        // Sends the map using a resized datatype to skip one parameter of the struct
        if((rc = MPI_Ssend(list_to_send, size, send_type, master, send_tag, comm)) != MPI_SUCCESS)
            return rc;

        free(list_to_send);
    }

    return rc;
}

int receiveMap(MapEntry **map, int size, int source, int tag, MPI_Datatype type, MPI_Comm comm) {
    int rc = 0;
    Couple *buf = malloc(sizeof(Couple) * size);
            
    if((rc = MPI_Recv(buf, size, type, source, tag, comm, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
        return rc;
    
    for(int i = 0; i < size; i++)
        increase_word_counter(map, buf[i].word, buf[i].counts);  
    free(buf);

    return rc;
}

int map_cmp(MapEntry *a, MapEntry *b) {
    return (b->counts - a->counts);
}

int create_csv(char *filename, MapEntry *map) {
    char file[FILENAME_SIZE] = {'\0'};
    FILE *fp;

    strcat(file, filename);
    strcat(file, ".csv");

    if((fp = fopen(file, "w")) == NULL) {
        fprintf(stderr, "Error: fopen error on csv file\n");
            
        return EXIT_FAILURE;
    }

    fprintf(fp, "Word,Frequency\n");

    for(MapEntry *e = map; e != NULL; e = e->hh.next) {
        fprintf(fp, "%s,%d\n", e->word, e->counts);
    } 

    fclose(fp);

    return 0;
}