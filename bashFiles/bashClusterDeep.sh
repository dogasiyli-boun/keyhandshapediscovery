#!/bin/bash

readonly curHost="$HOSTNAME"
readonly folderCode="/home/dg/PycharmProjects/keyhandshapediscovery/"
readonly folderVenv="/home/dg/anaconda3/envs/keyhandshapediscovery"

fc=0
fv="-"
if [ $curHost == "WsUbuntu05" ];
then
  fc="/home/dg/PycharmProjects/keyhandshapediscovery/"
  fv="/home/dg/anaconda3/envs/keyhandshapediscovery"
elif [ $curHost == "doga-MSISSD" ];
then
  fc="??/keyhandshapediscovery/"
  fv="??/envs/"
elif [ $curHost == "doga-msi-ubu" ];
then
  fc="??/keyhandshapediscovery/"
  fv="??/envs/"
else
  fc="??/keyhandshapediscovery/"
  fv="??/envs/"
fi

readonly folderCode=fc
readonly folderVenv=fv


cd ${folderCode}
pwd
chmod +x clusterDeep.py
source activate ${folderVenv}

funcNameToRun=$1

[ "$funcNameToRun" == "runForBaseClusterResults" ] && 
python -c"import projRelatedScripts as funcPRS; funcPRS.runForBaseClusterResults(randomSeed = 5, clusterModels = ['Kmeans', 'GMM_diag'])" || 
python ./clusterDeep.py --trainMode cosae --dataToUse skeleton --applyCorr 2 --pcaCount 64 --numOfSigns 11 --epochs 100 --appendEpochBinary 1 --randomSeed 5

