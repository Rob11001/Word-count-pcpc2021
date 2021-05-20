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
    <li><a href="#acknowledgements">Acknowledgements</a></li>
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

Tra i due si è scelto di optare per il secondo principalmente per evitare un conteggio preventivo delle parole da parte del nodo MASTER. Infatti, il MASTER nel secondo caso non ha bisogno di fare una lettura preventiva, ma può definire la partitioning usando semplicemente i metadati dei file presenti nella directory.\
Per mantenere le informazioni relative ai file letti e alla "sezione" di input assegnata a ciascuno dei processi sono state utilizzate due struct con relativa creazione dei MPI Datatype per l'invio mediante Scatter.


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
Come detto in precedenza, la scelta di effettuare due comunicazioni(operazioni "costose") è dovuta all'impossibilità da parte del nodo ricevente di poter conoscere a priori la dimensione della lista di nomi di file che gli sarà recapitata. Tale scelta, però, sarebbe facilmente evitabile supponendo che i file della directory siano posti nello stesso identico ordine per tutti i processi. In tal caso, infatti, sarebbe sufficiente una sola comunicazione. Nonostante ciò, è stata scelta la prima implementazione (più costosa) per avere una maggiore generalità a discapito ovviamente di una comunicazione in più.

TODO: descrivere il passo successivo per la computazione con codice\
TODO: descrivere processo di raccolta dei dati\
TODO: descrivere cvs\
TODO: descrizione della soluzione proposta con spezzoni di codice\
TODO: nella descrizione della soluzione ricordarsi il problema tra PACK e MPI_struct_type

<!-- UTILIZZO -->
# **Utilizzo**
## **Prerequisites**

TODO: DESCRIVERE utilizzo

<!-- NOTE SULL'IMPLEMENTAZIONE -->
# **Note sull'implementazione**

TODO: UTF-8\
TODO: uthash\
TODO: tipi derivati e pack and unpack

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
