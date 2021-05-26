#!/bin/bash
# Sends the source file to all ip addresses in the conf_file
# Usage: ./flood-files.sh conf_file source_file 

file=$1
shift
source=$1

# Reads the IP addresses from the file(first word in the line)
i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

# Executes scp
for ip in ${ips[@]}; do
  echo `scp ${source} pcpc@${ip}:` 
done
