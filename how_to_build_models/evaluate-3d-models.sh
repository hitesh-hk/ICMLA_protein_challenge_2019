#!/bin/bash

set -e

EVAL="./DISTFOLD/test/eval-using-tmscore.pl"

maxjobs=0
for pdb in `ls ./pdb/*.pdb`;
   do
	id=$(basename $pdb)
	id=${id%.*}
	echo $id
    perl $EVAL $pdb ./distfold-jobs/output-$id all header
    maxjobs=$((maxjobs+1))
    if [[ "$maxjobs" -gt 150 ]]; then
        exit 0
    fi
   done
