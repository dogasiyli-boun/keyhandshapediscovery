#!/bin/bash

readonly curHost="$HOSTNAME"
readonly funcNameToRun=$1
readonly epochCountToRun=$2
echo "Bash version ${BASH_VERSION}..."
echo "computer name is" "${curHost}"

python -c"import os; print(os.environ['VIRTUAL_ENV']) if 'VIRTUAL_ENV' in os.environ else print('not a virtual env')"

if [ $curHost == "WsUbuntu05" ];
then
  readonly folderCode="/home/dg/PycharmProjects/keyhandshapediscovery/"
  readonly folderVenv="/home/dg/anaconda3/envs/keyhandshapediscovery"
  source activate ${folderVenv}
elif [ $curHost == "doga-MSISSD" ];
then
  readonly folderCode="/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/"
  readonly folderVenv="/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/venv/bin/activate"
  source ${folderVenv}
elif [ $curHost == "doga-msi-ubu" ];
then
  readonly folderCode="/home/doga/GithUBuntU/keyhandshapediscovery/"
  readonly folderVenv="/home/doga/GithUBuntU/keyhandshapediscovery/venv/bin/activate"
  source ${folderVenv}
else
  fc="??/keyhandshapediscovery/"
  fv="??/envs/"
fi

cd $folderCode
echo "current directory = " pwd
echo "passed para  =" "$funcNameToRun"
python -c"import os; print('venv current = ', os.environ['VIRTUAL_ENV'])"

if [ "0" == "1" ];
then
  chmod +x clusterDeep.py
  [ "$funcNameToRun" == "runForBaseClusterResults" ] &&
  {
    python -c"import projRelatedScripts as funcPRS; funcPRS.runForBaseClusterResults(randomSeed = 5, clusterModels = ['Kmeans', 'GMM_diag'])"
  } ||
  {
    python ./clusterDeep.py --trainMode cosae --posterior_dim 32 --dataToUse skeleton --applyCorr 2 --pcaCount 64 --numOfSigns 11 --epochs 100 --appendEpochBinary 1 --randomSeed 5
  }
else
  declare -i i=0
  declare -a posterior_dim_array
  declare -a pcaCount_array
  declare -a rnnDataMode_array
  applyCorr=0
  numOfSigns=41
  randomSeed=5
  appendEpochBinary=0
  posterior_dim_array=(32 64 128 256)
  rnnDataMode_array=(lookBack patchPerVideo frameOverlap)
  epochs=$epochCountToRun
  for trainMode in sae rsa
  do
    for dataToUse in skeleton hog sn
    do
      [ "$dataToUse" == "skeleton" ] &&
      {
        pcaCount_array=(-1 32 64 96)
      } ||
      {
        pcaCount_array=(-1 256 512 1024)
      }
      for pcaCount in ${pcaCount_array[@]}
      do
        for posterior_dim in ${posterior_dim_array[@]}
        do
          [ "$trainMode" == "sae" ] &&
          {
            echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr
            python ./clusterDeep.py --trainMode $trainMode --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
          } ||
          {
            for rnnDataMode in ${rnnDataMode_array[@]}
            do
              echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr $rnnDataMode
              python ./clusterDeep.py --trainMode $trainMode --rnnDataMode $rnnDataMode --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
            done
          }

          i=$((i+1))
        done
      done
    done
  done
fi