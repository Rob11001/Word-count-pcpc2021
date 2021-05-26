<!-- Title -->

# WORD COUNT
|**Word count**|**Bruno Roberto**| 
|---|---|
<br>

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary><h1 style="display: inline-block">Table of Contents</h1></summary>
  <ol>
    <li>
      <a href="#descrizione-del-problema">Descrizione del problema</a>
    </li>
    <li>
      <a href="#soluzione-proposta">Soluzione proposta</a>
    </li>
    <li>
      <a href="#correttezza-della-soluzione-proposta">Correttezza della soluzione proposta</a>
    </li>
    <li><a href="#utilizzo">Utilizzo</a></li>
    <li><a href="#note-sullimplementazione">Note sull'implementazione</a></li>
    <li><a href="#benchmark">Benchmark</a>
      <ul>
        <li><a href="#scalabilità-forte">Scalabilità forte</a></li>
        <li><a href="#scalabilità-debole">Scalabilità debole</a></li>
        <li><a href="#descrizione-dei-risultati">Descrizione dei risultati</a></li>
      </ul>
    </li>
    <li><a href="#conclusioni">Conclusioni</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<br>

<!-- DESCRIZIONE DEL PROBLEMA -->

# **Descrizione del problema**
Il **Word count** è un "semplice" problema che consiste nel conteggio del numero di parole presenti in un documento o una porzione di testo.\
Il conteggio delle parole è richiesto in molte applicazioni pratiche che spaziano dal giornalismo, alle procedure legali, agli annunci pubblicitari e molto altro. Proprio per tale motivo la mole di dati da dover processare più facilmente e velocemente diventare "molto" grande e ciò ci conduce alla necessità di un approccio distribuito.\
Il seguente progetto propone, dunque, una versione distribuita di Word count mediante l'uso di MPI.

<!-- SOLUZIONE PROPOSTA -->

# **Soluzione proposta**
Andiamo a descrivere la soluzione proposta per la risoluzione di Word count evidenziando le problematiche riscontrate ed esponendo quali sono state le scelte implementative. 
</br>
</br>
Innanzitutto la prima problematica affrontata è stata quella della distribuzione del carico di lavoro tra i processi coinvolti nella computazione.\
Seguendo la traccia del problema il nodo MASTER deve suddividere i file da processare tra i vari nodi. Tale suddivisione non può essere condotta basandosi solo sul numero dei file altrimenti risulterebbe non omogenea.\
Bisogna, dunque, tener conto del contenuto di ciascun file, e ci sono due possibili approcci per dare un peso ai file:
- usare il numero di parole contenute nel file
- usare la dimensione del file in bytes

Tra i due si è scelto il secondo principalmente per evitare un conteggio preventivo delle parole da parte del nodo MASTER. Infatti, il MASTER nel secondo caso non ha bisogno di fare una lettura preventiva, ma può definire la partitioning usando semplicemente i metadati dei file presenti nella directory.\
Per mantenere le informazioni relative ai file letti e alla "sezione" di input assegnata a ciascuno dei processi sono state utilizzate due struct con relativa creazione dei MPI Datatype associati per l'invio mediante Scatter.


Di seguito è riportata la definizione delle struct usate e l'inizializzazione della comunicazione con la creazione dei tipi di dato derivato:
```c
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
```

```c
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
        fprintf(stderr, "Error : you have to pass a directory path\n");

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

    // FileInfo struct type
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
```

Come visibile dallo snippet precedente, oltre alla creazione dei _MPI_Datatype_, si è scelto di creare anche un nuovo _Communicator_ con una topologia a grafo per riordinare i nodi in maniera da ottimizzare la comunicazione. La topologia scelta rispecchia banalmente la comunicazione MASTER-SLAVE tra i nodi ponendo al centro del grafo il MASTER.

Lo snippet seguente mostra, invece, la funzione utilizzata per effettuare il partitioning dei file. Per ognuno dei processi, MASTER incluso, viene calcolato il numero di file che dovrà processare, l'offset da cui dovrà partire nel primo file e l'offset dopo il quale dovrà fermarsi nell'ultimo file. In tal modo è visibile come ogni processo abbia la stessa quantità di dati da processare, realizzando una distribuzione abbastanza equa del lavoro.

```c
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

```
Una volta effettuato il partitioning, ovviamente il MASTER necessita di comunicare ad ogni processo la lista dei file assegnatigli e la "sezione" di input su cui dovrà lavorare. Per realizzare ciò, non potendo un processo conoscere a priori la lunghezza della lista di file che gli verrà inoltrata dal MASTER, è stato necessario eseguire due step:
1. una prima comunicazione per fornire ad ogni processo il numero di file che riceverà e gli offset di partenza e fine computazione
2. una seconda comunicazione per poter inviare ad ogni processo la lista dei nomi dei file che dovrà processare

Tale comunicazione, come visibile nello snippet seguente, è stata realizzata attraverso due semplici comunicazioni collettive scatter, usando come tipi di dati quelli definiti precedentemente per le struct.

```c
    // Processes indexes 
    if((rc = MPI_Scatter(pindexes, 1, process_data, &pindex, 1, process_data, master, graph_comm)) != MPI_SUCCESS)
        return rc;
    
    FileInfo *recvfiles = malloc(sizeof(FileInfo) * pindex.numfiles);
    
    // Files info
    if((rc = MPI_Scatterv(files, send_counts, displs, file_info, recvfiles, pindex.numfiles, file_info, master, graph_comm))!= MPI_SUCCESS)
        return rc;
```
Come detto in precedenza, la scelta di effettuare due comunicazioni(operazioni "costose") è dovuta all'impossibilità da parte del nodo ricevente di poter conoscere a priori la dimensione della lista dei nomi di file che gli sarà recapitata. Tale scelta, però, sarebbe facilmente evitabile supponendo che i file della directory siano posti nello stesso identico ordine per tutti i processi. In tal caso, infatti, sarebbe sufficiente una sola comunicazione. Nonostante ciò, la scelta finale è ricaduta sulla prima implementazione (più costosa) per avere una maggiore generalità a discapito ovviamente di una comunicazione in più.

A questo punto ogni processo possiede tutti i dati necessari a portare avanti il suo task, ovvero effettuare il conteggio delle parole sulla porzione di dati assegnatagli, e può iniziare dunque la computazione parallela.\
Tale computazione è realizzata facendo sì che ogni processo legga ad uno ad uno i file assegnatigli, eseguendo se necessario una _seek_ per posizionarsi correttamente. L'identificazione delle parole durante la lettura del file è semplicemente implementata mediante lo scorrimmento carattere per carattere del buffer di appoggio usato per la lettura stessa. Di seguito uno snippet di codice contenente la funzione usata per la computazione.

```c
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
```
Come visibile nella funzione, a causa della strategia di partitioning dei file scelta è stato necessario gestire il conflitto di parole "condivise" tra due processi successivi, ovvero parole poste esattamente a cavallo degli input di due processi successivi. Per gestire tale problemtica si è scelto, per evitare comunicazioni superflue, di far banalmente processare la parola al primo dei due processi, mentre il secondo si limita semplicemente a riconoscere ed ignorare tale parola "condivisa".

Per il mantenimento delle informazioni relative alla frequenza delle parole, invece, si è scelto di utilizzare una semplice Hash Table, usando come chiavi le parole stesse (in lower-case). La scelta di usare un Hash Table è stata guidata dalla necessità di poter usufruire di una struttura dati che permettesse in maniera efficiente e veloce l'aggiornamento delle frequenze.
Nello snippet seguente la definizione della struct del singolo elemento della Hash Table e le funzioni di utilities usate per l'aggiornamento della frequenza (e la gestione della codifica UTF-8 dei file) durante la computazione (si è utilizzata l'Hash Table fornita dalla "libreria" **uthash.h**):

```c
/* An entry of the Hash. It's a couple key-value. The key is a string, the value an integer */
typedef struct {
    char* word;           /* key */
    int frequency;
    UT_hash_handle hh;  /* makes this structure hashable */ 
} MapEntry;

...
/**
 * @brief Adds the key to the hash with the passed value
 * 
 * @param map the hash
 * @param word the key
 * @param counts the value 
 */
void add_word(MapEntry **map, char* word_str, int counts) {
    MapEntry *s = malloc(sizeof(MapEntry));
    
    s->word = strdup(word_str);
    s->frequency = counts;

    HASH_ADD_STR(*map, word, s);
}

/**
 * @brief Increases the value of the entry with the passed key, or creates it if necessary
 * 
 * @param map the hash
 * @param word the key
 * @param counts the value to add to old value
 */
void increase_word_counter(MapEntry **map, char *word, int counts) {
    MapEntry *entry = NULL;
    
    HASH_FIND_STR(*map, word, entry);
    if(entry != NULL) 
        entry->frequency += counts;
     else 
        add_word(map, word, counts);
}

/**
 * @brief Calculates the number of bytes of the character using UTF-8
 * 
 * @param first_char char
 * @return int value between 1 and 4
 */
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

/**
 * @brief Checks if is a symbol
 * 
 * @param ch 
 * @return [0-1]
 */
int issymbol(char ch) {
    return ch == '\'' || ch == '-';
}

/**
 * @brief Checks is is a symbol
 * 
 * @param ch 
 * @return [0-1]
 */
int ismulticharsymbol(char *ch) {
    return strcmp(ch, "”") == 0 || strcmp(ch, "—") == 0 || strcmp(ch, "“") == 0 || strcmp(ch, "«") == 0 || strcmp(ch, "»") == 0;
}
```

Completata la computazione, ogni processo possiede una propria Hash Table locale che deve inoltrare al MASTER in maniera tale che sia eseguito il merge delle frequenze ottenendo l'istogramma globale della directory in input(corrisponde all'operazione di Reduce). L'operazione di gathering è stata eseguita attraverso una serie di comunicazioni non-bloccanti in maniera tale da permettere al nodo MASTER di effettuare il merge dei dati nella sua Hash Table "globale" man mano che i dati gli vengono inoltrati, senza dover aspettare tutti i processi come invece sarebbe stato necessario se si fosse scelto di utilizzare una comunicazione collettiva bloccante come MPI_Gather.

Una problematica importante affrontata durante il processo di raccolta dati è stata relativa all'invio dei dati. In un primo momento si è pensato di procedere usando, come nei casi precedenti, un MPI_Datatype della struct contenente la coppia parola-frequenza, però ciò richiedeva che la dimensione del campo `word` non potesse essere allocata dinamicamente, ma dovesse essere definita a priori. Questo vincolo comportava di conseguenza uno spreco di dati non banale. Ad esempio, supponiamo che la dimensione di word fosse 100, per parole tipo "dog", "the", "cat", ovvero molto brevi (e anche molto comuni), sarebbe stato comunque necessario inviare tutto array (invio di quasi un centinaio di byte "inutili"). Dunque, per evitare tale spreco di banda, si è deciso di adottare una soluzione più a "basso livello" mediante `MPI_Pack` and `MPI_Unpack`. L'idea è stata quella di compattare in un unico array di byte le coppie parola-frequenza per scompattarle poi a destinazione, trasmettendo solo i byte strettamente necessari. L'oggetto `MPI_Packed` da inviare è stato realizzato compattando le word nel seguente modo: "parola_1-frequenza_1-parola_2-frequenza_2-...".\
L'operazione duale di scompattamento, invece, sfrutta il fatto che ogni parola termina con il carattere '`\0`' per ricreare nuovamente le coppie parola-frequenza. 

In entrambi i casi non potendo conoscere a priori la dimensione delle Hash Table locali, è stato necessario eseguire una prima comunicazione preventiva.\
Di seguito lo snippet di codice delle funzioni utilizzate per la raccolta e riduzione dei dati.

```c
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
int gatherAndReduce(MapEntry **master_map, int master, MapEntry **local_map, int rank, int numtasks, int size_tag, int send_tag, MPI_Comm comm) {
    int rc = 0;
    
    if(rank == master) {
        // Request and sizes
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

            increase_word_counter(master_map, e->word, e->frequency);

            // Checks if some process has sent its map
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
            if((rc = MPI_Pack(&(e->frequency), 1, MPI_INT, buf, buf_size, &pos, comm)) != MPI_SUCCESS)
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
```
 Arrivati a questo punto ogni processo ha completato il suo lavoro e il MASTER possiede l'istogramma globale dell'input. Dunque, non gli rimane che effettuare l'ordinamento della struttura in base alla frequenza, ed infine scrivere l'output in un file csv.\
Di seguito lo snippet che esegue tali operazioni conclusive.

```c
    if(rank == master) {
        // Sorting
        HASH_SORT(master_map, map_cmp);
        
        #ifdef DEBUG
        int tot = 0;
        for(MapEntry *e = master_map; e != NULL; e = e->hh.next) 
            tot += e->frequency;
        printf("Unique-words: %d, Total words: %d\n", HASH_COUNT(master_map), tot);
        #endif      
        
        // Csv creation
        if((rc = create_csv(argv[1], master_map)) != 0) {
            fprintf(stderr, "Error in csv file creation\n");
        
            return rc;
        }

        printf("Completed '%s.csv' file creation...\n", argv[1]);

    }
```

```c
/**
 * @brief Compares two Map Entry using counts 
 * 
 * @param a Map entry
 * @param b Map entry
 * @return int 0 if are equals, > 0 if a is less frequent than b, < 0 otherwise
 */
int map_cmp(MapEntry *a, MapEntry *b) {
    return (b->frequency - a->frequency);
}

/**
 * @brief Creates a csv file
 * 
 * @param filename name of the file
 * @param map the map which will be written in the file
 * @return int 0 if it's all okay, non-zero otherwise
 */
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
        fprintf(fp, "%s,%d\n", e->word, e->frequency);
    } 

    fclose(fp);

    return 0;
}
```

<!-- CORRETTEZZA -->

# **Correttezza della soluzione proposta**
La correttezza della soluzione proposta non è stata dimostrata formalmente data la difficoltà di eseguire una dimostrazione simile, ma mediante l'esecuzione di una serie di test case molto semplici, che andassero a coprire i vari casi possibili.\
Tali test case sono presenti nella directory *simple_test* e il file csv rappresentante il file di output dell'oracolo è presente nella directory *simple_test_oracle_result*.\
L'esecuzione dei test ovviamente è stata eseguita analizzando il comportamento della soluzione anche al variare del numero di processi, a partire dal semplice caso sequenziale con un solo processo, e l'output fornito risultava essere sempre il medesimo a dimostrazione della correttezza.\
Di seguito, come esempio, due screen di output: il primo sequenziale e il secondo parallelo con 3 processi.

<p float="left">
    <img src="images/sequential.png" alt="sequential" width="150" height="200">
    <img src="images/parallel.png" alt="parallel" width="150" height="200">
</p>

<!-- UTILIZZO -->

# **Utilizzo**

## **Prerequisites**
- ambiente con OpenMPI

Innanzitutto effettuare una copia locale del progetto, o almeno del file **"word_count.c"** e della directory **"libs"**.\
Porre entrambi nella stessa directory o in caso contrario bisognerà cambiare manualmente 

```c
#include "./libs/uthash.h"
```
nel file C per rispettare il nuovo path.

Fatto ciò, per compilare il codice basterà eseguire il comando standard di compilazione di ``mpicc``:
```c
mpicc word_count.c -o word_count
```
Per eseguire il programma su un cluster di macchine invece basterà utilizzare il seguente comando:
```c
mpirun -np x --hostfile hf word_count dir_path
```
specificando con `-np` il numero di processi, con `--hostfile` il file contenente le macchine del cluster con relativo numero di vCPU.\
**Attenzione**: il parametro da linea di comando `dir_path` indica il path della directory in cui sono presenti i file su cui sarà eseguito il programma. Dunque, è estremamente importante che la directory e i file siano presenti nel medesimo path per tutti i nodi coinvolti nella computazione (semplificazione del problema).

<!-- NOTE SULL'IMPLEMENTAZIONE -->

# **Note sull'implementazione**

Nella gestione della codifica dei file è stata eseguita una semplificazione ragionevole supponendo che tutti i file siano in una unica codifica, **UTF-8**. Questo ha permesso una gestione più facile dei caratteri speciali composti da multipli byte.\
Inoltre, le parole sono state individuate usando come caratteri di terminazione caratteri di *whitespace* come '`\n`','` `' e altri, come visibile negli snippet precedenti. È stato poi data la possibilità di riconoscere anche parole con la presenza di al più un simbolo come '`'`' o '`-`', permettendo di riconoscere, come singole parole, parole come "cat-walk" o "D'Antonio".

Per la memorizzazione delle coppie parola-frequenza è stata usata una semplice Hash Table realizzata mediante la "libreria" [`uthash.h`](http://troydhanson.github.io/uthash/userguide.html#_a_hash_in_c) ([link alla pagina github](https://github.com/troydhanson/uthash)).

Per la comunicazione dei dati tra il MASTER e gli altri processi, come già detto nella sezione precedente, sono stati utilizzati sia i tipi MPI derivati attraverso `MPI_Type_create_struct` (permettendo l'invio e la ricevione di struct) che il packing e l'unpacking dei dati mediante `MPI_Pack` e `MPI_Unpack`. La scelta del secondo metod, in particoalre, è stata introdotta dalla necessità di ridurre la quantità di dati "inutili" da trasmettere nel processo di gathering finale, riuscendo così a compattare in un unico buffer contiguo solo i dati strettamente necessari, e di conseguenza inviarli tutti insieme.

Rimanendo sempre nella tematica della comunicazione sono stati utilizzati sia meccanismi di comunicazione collettiva bloccanti per la distribuzione dei dati, come `MPI_Scatter` e `MPI_Scatterv`, e sia meccanismi di comunicazione non bloccanti, `MPI_IRecv`, per la raccolta dei dati, in modo da permettere alla fasi di computazione e comunicazione di interfogliarsi.

TODO: topologia?

<!-- BENCHMARK -->

# **Benchmark**
## **Scalabilità forte**

## **Scalabilità debole**

## **Descrizione dei risultati**

<!-- CONCLUSIONI -->

# **Conclusioni**

<!-- LICENSE -->

# **License**

Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
