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
#define DEBUG

// Structures

/* Contains the required info for describing a file*/
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

/* An entry of the Hash. It's a couple key-value. The key is a string, the value an integer */
typedef struct {
    char* word;           /* key */
    int counts;
    UT_hash_handle hh;  /* makes this structure hashable */ 
} MapEntry;

// Utility functions

/**
 * @brief Reads all files in passed directory and creates a FileInfo foreach of them
 * 
 * @param dir_name Directory path (absolute path)
 * @param total_size_in_bytes it'll contain the total size of the read files in bytes
 * @param numfiles It'll contain the number of files read in the directory
 * @return FileInfo* The array of FileInfo, or NULL if occurs an error
 */
FileInfo *dir_info_extraction(char *dir_name, off_t *total_size_in_bytes, int *numfiles);

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
int compute(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path, int rank);

/**
 * @brief All processes send their data to the master, then the master collects them and reduces them in
 * only one hash
 * 
 * @param master_map the hash used to collect all data (only used by the master)
 * @param master the rank of the master
 * @param local_map the map to send to the master
 * @param rank the rank of the process
 * @param numtasks the total number of processes in the comm
 * @param size_tag tag used for the messages which processes use to specify the size of the data to send
 * @param send_tag tag used to send the real data
 * @param comm the communicator used
 * @return int 0 if it's all okay, non-zero otherwise
 */
int gatheringAndReduce(MapEntry **master_map, int master, MapEntry **local_map, int rank, int numtasks, int size_tag, int send_tag, MPI_Comm comm);

/**
 * @brief Utility function which handles the receive of a list of couple from a process
 * 
 * @param map the hash in which the couples will be added
 * @param size the size of the packed message
 * @param source the process from which receives the data
 * @param tag the tag used in the communication
 * @param comm the communicator used
 * @return int 0 if it's all okay, non-zero otherwise
 */
int receiveMap(MapEntry **map, int size, int source, int tag, MPI_Comm comm);

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
    MPI_Datatype process_data, file_info;
    MPI_Aint extent, lb;
    int blocklengths[3];
    MPI_Datatype types[3];
    MPI_Aint displacements[3];
    ProcessIndex pindex;
    MapEntry *local_map = NULL;
    MPI_Group graph_group, world_group;
    MPI_Comm graph_comm;

    // Master only
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
        // Read files's info from directory
        off_t total_size_in_bytes = 0;
        int numfiles = 0;
        
        if((files = dir_info_extraction(argv[1], &total_size_in_bytes, &numfiles)) == NULL) {
            fprintf(stderr, "Reading error on directory '%s'\n", argv[1]);

            return EXIT_FAILURE;
        }
        
        #ifdef DEBUG
        // For debugging purpose
        batch_size = total_size_in_bytes / numtasks;
        remainder = total_size_in_bytes % batch_size;
        printf("File read: %d\n", numfiles);
        printf("Size in bytes: %ld\n", total_size_in_bytes);
        printf("batch_size: %ld\n", batch_size);
        #endif

        // Files schedulation
        send_counts = malloc(sizeof(int) * numtasks);
        displs = malloc(sizeof(int) * numtasks);
        pindexes = malloc(sizeof(ProcessIndex) * numtasks);
        
        file_scheduling(numtasks, send_counts, displs, pindexes, total_size_in_bytes, numfiles, files);

        #ifdef DEBUG
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
        #endif   
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
    if((rc = compute(recvfiles, pindex.numfiles, pindex.start_offset, pindex.end_offset, &local_map, argv[1], rank)) != 0) {
        fprintf(stderr, "Computation error, error code: %d\n", rc);

        return EXIT_FAILURE;
    }
    
    #ifdef DEBUG
    printf("Task %d -> total_word: %d\n", rank, HASH_COUNT(local_map));
    #endif

    // Gathering and Reduce
    if((rc = gatheringAndReduce(&master_map, master, &local_map, rank, numtasks, count_tag, send_tag, graph_comm)) != MPI_SUCCESS)
        return rc;

    if(rank == master) {
        // Sorting
        HASH_SORT(master_map, map_cmp);
        
        #ifdef DEBUG
        int tot = 0;
        for(MapEntry *e = master_map; e != NULL; e = e->hh.next) 
            tot += e->counts;
        printf("Unique-words: %d, Total words: %d\n", HASH_COUNT(master_map), tot);
        #endif      
        
        // Csv creation
        if((rc = create_csv(argv[1], master_map)) != 0) {
            fprintf(stderr, "Error in csv file creation\n");
        
            return rc;
        }

        printf("Completed '%s.csv' file creation...\n", argv[1]);

    }

    /* ************************************************* */
    MPI_Barrier(graph_comm);
    end = MPI_Wtime();
    fflush(stdout);

    MPI_Type_free(&process_data);
    MPI_Type_free(&file_info);
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
            free(e->word);
            next = e->hh.next;
            free(e);
        }

    }

    for(MapEntry *e = local_map, *next; e != NULL; e = next) {
        free(e->word);
        next = e->hh.next;
        free(e);
    }
    
    return EXIT_SUCCESS;
}


// Functions

FileInfo *dir_info_extraction(char *dir_name, off_t *total_size_in_bytes, int *numfiles) {
    DIR* FD;
    struct dirent* in_file;
    struct stat buf;
    FileInfo *files = NULL;

    if((FD = opendir(dir_name)) == NULL) {
            fprintf(stderr, "Error : Failed to open input directory\n");

            return NULL;
    }
    
    *total_size_in_bytes = 0;
    files = malloc(sizeof(FileInfo));
    *numfiles = 0;
    int size = 0;
    off_t total_size = 0;
    // FileInfo reading (directory reading)
    while((in_file = readdir(FD))) {
        char file[FILENAME_SIZE] = {'\0'};
        
        if(in_file->d_type != DT_REG)
            continue;
        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))    
            continue;            
        size += 1;
        files = realloc(files, sizeof(FileInfo) * size);
    
        strcat(file, dir_name);
        strcat(file, "/");
        if(stat(strcat(file, in_file->d_name), &buf) != 0)
            fprintf(stderr, "Error: Stat error on file %s\n", in_file->d_name);
        
        files[size - 1].filename[0] = '\0';
        strcat(files[size - 1].filename, in_file->d_name);
        files[size - 1].size_in_bytes = buf.st_size;
        total_size += buf.st_size;
        
        #ifdef DEBUG
        // For debugging purpose
        printf("%s - %ld\n", in_file->d_name, buf.st_size);
        #endif
    }

    *numfiles = size;
    *total_size_in_bytes = total_size;

    return size > 0 ? files : NULL;
}



void file_scheduling(int numtasks, int *send_counts, int *displs, ProcessIndex *pindexes, int total_size, int numfiles, FileInfo *files) {
    int batch_size = total_size / numtasks;
    int remainder = total_size % batch_size;
    off_t next = 0;
    off_t nextfile_size = files[0].size_in_bytes;

    // Files'schedulation
    for(int i = 0, j = 0; i < numtasks; i++) { // i - processes index, j - files index
        pindexes[i].start_offset = next;  // Process i beginning offset
        displs[i] = j;                    // Process i beginning file
        send_counts[i] = 0;
        int mybatch = batch_size + ((remainder > i));   // Bytes of process i
        
        while(j < numfiles && mybatch > 0) {
            send_counts[i] += 1;            // Assigns the file j to processor i
            
            if(mybatch <= nextfile_size) {       
                pindexes[i].end_offset = files[j].size_in_bytes - nextfile_size + mybatch; // The position in which I've stopped in the last file 
                next = (mybatch == nextfile_size) ? 0 : pindexes[i].end_offset + 1;       // Starting offset for the next processor
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
    
    s->word = strdup(word_str);
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

int compute(FileInfo *files, int num_files, long start_offset, long end_offset, MapEntry **map, char *dir_path, int rank) {
    FILE *fp;
    char filename[FILENAME_SIZE];
    char readbuf[READ_BUF] = {'\0'};
    long offset = start_offset, rd = 0;

    // Foreach file
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
        int current_word_size = 0, jump = 0;
        
        // First reading out of the loop 
        char *p = fgets(readbuf, READ_BUF, fp);
    
        // To handle conflicts on word between processes
        if(p != NULL && i == 0 && offset > 0) {  
            rd -= 1;
            
            // Skips whitespaces and the word shared by the two processes, if exists
            while((p - readbuf) < READ_BUF && (isalpha(*p) || *p < 0 || issymbol(*p))) { // checks also if it's a character UTF-8 with more bytes or a symbol
                p += 1;
                rd += 1;
            }
            
            // To set the correct rd and p value
            if(rd < 0) {    
                rd = 0;
                p += 1;
            }
        }

        // File's reading
        while(p != NULL && !jump) {  // jump is a boolean variable to skip the loop  
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
                    // I found a word
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

            // Buffer reset
            readbuf[0] = '\0';  
            // New buffer's reading
            p = fgets(readbuf, READ_BUF, fp);
        }

        // If I ended to read the file and there is still a word in currentWord buffer
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

int gatheringAndReduce(MapEntry **master_map, int master, MapEntry **local_map, int rank, int numtasks, int size_tag, int send_tag, MPI_Comm comm) {
    int rc = 0;
    
    if(rank == master) {
        // Request and counts
        MPI_Request *reqs = malloc(sizeof(MPI_Request) * (numtasks - 1));
        int *bufsizes = malloc(sizeof(int) * (numtasks - 1));

        // Post for size
        for(int p = 0, i = 0; p < numtasks; p++) {
            if(p == master)
                continue;
            
            if((rc = MPI_Irecv(bufsizes + i, 1, MPI_INT, p, size_tag, comm, reqs + i)) != MPI_SUCCESS)
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
                if((rc = receiveMap(master_map, bufsizes[index], (index >= master) ? index + 1 : index, send_tag, comm)) != MPI_SUCCESS)
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
            if((rc = receiveMap(master_map, bufsizes[index], (index >= master) ? index + 1 : index, send_tag, comm)) != MPI_SUCCESS)
                return rc;

            received += 1; // Updates counter
        }
        
        free(reqs);
        free(bufsizes);

    } else {
        int size = HASH_COUNT(*local_map), pos = 0, buf_size = 0;
        char *buf = malloc(sizeof(char));

        // Packing of own Hashmap
        for(MapEntry *e = *local_map; e != NULL; e = e->hh.next) {
            int str_size = strlen(e->word) + 1;
            buf_size += str_size + sizeof(int);
            buf = realloc(buf, buf_size * sizeof(char));    // Updates bufsize 

            // Packs string and integer
            if((rc = MPI_Pack(e->word, str_size, MPI_CHAR, buf, buf_size, &pos, comm)) != MPI_SUCCESS)
                return rc;
            if((rc = MPI_Pack(&(e->counts), 1, MPI_INT, buf, buf_size, &pos, comm)) != MPI_SUCCESS)
                return rc;
        }
        // Sends the size of the packed map
        if((rc = MPI_Send(&buf_size, 1, MPI_INT, master, size_tag, comm)) != MPI_SUCCESS)
            return rc;
        // Sends the map using MPI_PACKED datatype
        if((rc = MPI_Ssend(buf, buf_size, MPI_PACKED, master, send_tag, comm)) != MPI_SUCCESS)
            return rc;

        free(buf);
    }

    return rc;
}

int receiveMap(MapEntry **map, int size, int source, int tag, MPI_Comm comm) {
    int rc = 0;
    char *buf = malloc(size), *p;
    int frequency, pos = 0;

    // Receives packed buffer
    if((rc = MPI_Recv(buf, size, MPI_PACKED, source, tag, comm, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
        return rc;

    // Unpacks the buffer and reads all couples word-frequency
    while(pos < size) {
        int str_size = strlen(buf + pos) + 1;
        p = malloc(sizeof(char) * str_size);

        // Unpacks the word
        if((rc = MPI_Unpack(buf, size, &pos, p, str_size, MPI_CHAR, comm)) != MPI_SUCCESS)
            return rc;
        // Unpacks the frequency
        if((rc = MPI_Unpack(buf, size, &pos, &frequency, 1, MPI_INT, comm)) != MPI_SUCCESS)
            return rc;
        
        increase_word_counter(map, p, frequency);  
        free(p);     
    }

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