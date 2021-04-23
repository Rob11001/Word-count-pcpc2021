#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    DIR* FD;
    struct dirent* in_file;
    struct stat buf;

    if((FD = opendir(argv[1])) == NULL) {
        fprintf(stderr, "Error : Failed to open input directory\n");

        return EXIT_FAILURE;
    }
    
    while((in_file = readdir(FD))) {
        char file[256] = {'\0'};

        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))    
            continue;
        
        strcat(file, argv[1]);
        strcat(file, "/");

        if(stat(strcat(file, in_file->d_name), &buf) != 0)
            printf("Err");
        printf("%s - %ld\n", in_file->d_name, buf.st_size);
    }

    return EXIT_SUCCESS;
}