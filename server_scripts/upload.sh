#!/bin/bash

user_name=andrewhz  # Change username
local_prefix='/c/Users/electron/etrap'  # Change these to the desired prefix
server_prefix='etrap/data'



# No need to change below
help_info='>>> SYNTAX: "sh upload.sh [-r] [filename]", use [-r] if uploading folder'

source_prefix=${local_prefix}
destination_prefix=${server_prefix}
copy_folder=false
while getopts "r" opt; do
	case ${opt} in 
		r)
			copy_folder=true
			;;
		\?)
			echo ${help_info}
			exit 0
			;;
	esac
done

if [ "$copy_folder" == "true" ]
then
	echo ">>> Uploading folder ${2} from LOCAL:${source_prefix} to SERVER:${destination_prefix}"
	scp -r ${source_prefix}/${2} ${user_name}@dtn.brc.berkeley.edu:${destination_prefix}
else
	echo ">>> Uploading file ${1} from LOCAL:${source_prefix} to SERVER:${destination_prefix}"
	scp ${source_prefix}/${1} ${user_name}@dtn.brc.berkeley.edu:${destination_prefix}
fi


