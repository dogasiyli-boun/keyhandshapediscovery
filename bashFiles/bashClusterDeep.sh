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
  echo "curEnv=" $CONDA_PREFIX
elif [ $curHost == "doga-MSISSD" ];
then
  readonly folderCode="/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/"
  readonly folderVenv="/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/venv/bin/activate"
  source ${folderVenv}
  python -c"import os; print(os.environ['VIRTUAL_ENV']) if 'VIRTUAL_ENV' in os.environ else print('not a virtual env')"
elif [ $curHost == "doga-msi-ubu" ];
then
  readonly folderCode="/home/doga/GithUBuntU/keyhandshapediscovery/"
  readonly folderVenv="/home/doga/GithUBuntU/keyhandshapediscovery/venv/bin/activate"
  source ${folderVenv}
  python -c"import os; print(os.environ['VIRTUAL_ENV']) if 'VIRTUAL_ENV' in os.environ else print('not a virtual env')"
else
  fc="??/keyhandshapediscovery/"
  fv="??/envs/"
fi

cd $folderCode
echo "current directory = " pwd
echo "funcNameToRun=" "$funcNameToRun" ",epochCountToRun=" "$epochCountToRun"

if [ "$funcNameToRun" == "runForBaseClusterResults" ];
then
  chmod +x clusterDeep.py
  [ "$funcNameToRun" == "runForBaseClusterResults" ] &&
  {
    echo here111
    python -c"import projRelatedScripts as funcPRS; funcPRS.runForBaseClusterResults(randomSeed = 5, clusterModels = ['Kmeans', 'GMM_diag', 'GMM_full', 'Spectral'])"
  } ||
  {
    echo here222
    python ./clusterDeep.py --trainMode cosae --posterior_dim 32 --dataToUse skeleton --applyCorr 2 --pcaCount 64 --numOfSigns 11 --epochs 100 --appendEpochBinary 1 --randomSeed 5
  }
elif [ $curHost == "doga-MSISSD" ];
then
  declare -i i=0
  declare -a posterior_dim_array
  declare -a pcaCount_array
  declare -a rnnDataMode_array
  declare -a numOfSigns_array
  applyCorr=2
  numOfSigns_array=(11 41)
  randomSeed=5
  appendEpochBinary=0
  posterior_dim_array=(256 512) # 32 64 128
  rnnDataMode_array=(lookBack) # patchPerVideo frameOverlap
  epochs=$epochCountToRun
  for numOfSigns in ${numOfSigns_array[@]}
  do
    for trainMode in cosae corsa
    do
      for dataToUse in skeleton hog sn
      do
        [ "$dataToUse" == "skeleton" ] &&
        {
          pcaCount_array=(32 64 96) #64 96
        } ||
        {
          pcaCount_array=(256 512) #512 1024
        }
        for pcaCount in ${pcaCount_array[@]}
        do
          for posterior_dim in ${posterior_dim_array[@]}
          do
            [ "$trainMode" == "cosae" ] &&
            {
              echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr $numOfSigns
              python ./clusterDeep.py --trainMode $trainMode --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
            } ||
            {
              for rnnDataMode in ${rnnDataMode_array[@]}
              do
                echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr $rnnDataMode $numOfSigns
                python ./clusterDeep.py --trainMode $trainMode --rnnDataMode $rnnDataMode --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
              done
            }
            i=$((i+1))
          done
        done
      done
    done
  done
else
  declare -i i=0
  declare -a posterior_dim_array
  declare -a pcaCount_array
  declare -a rnnDataMode_array
  declare -a numOfSigns_array
  applyCorr=2
  numOfSigns_array=(11 41)
  randomSeed=5
  appendEpochBinary=0
  posterior_dim_array=(256 512) # 64 256
  rnnDataMode_array=(lookBack) # patchPerVideo frameOverlap
  epochs=$epochCountToRun
  for numOfSigns in ${numOfSigns_array[@]}
  do
    for trainMode in cosae corsa
    do
      for dataToUse in skeleton hog sn
      do
        [ "$dataToUse" == "skeleton" ] &&
        {
          pcaCount_array=(32 64 96) #64 96
        } ||
        {
          pcaCount_array=(256 512) #512 1024
        }
        for pcaCount in ${pcaCount_array[@]}
        do
          for posterior_dim in ${posterior_dim_array[@]}
          do
            [ "$trainMode" == "cosae" ] &&
            {
              echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr $numOfSigns
              python ./clusterDeep.py --trainMode $trainMode --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
            } ||
            {
              for rnnDataMode in ${rnnDataMode_array[@]}
              do
                echo $i $trainMode $dataToUse $pcaCount $posterior_dim $applyCorr $rnnDataMode $numOfSigns
                python ./clusterDeep.py --trainMode $trainMode --rnnDataMode $rnnDataMode --rnnTimesteps 1 --posterior_dim $posterior_dim --dataToUse $dataToUse --applyCorr $applyCorr --pcaCount $pcaCount --numOfSigns $numOfSigns --epochs $epochs --appendEpochBinary $appendEpochBinary --randomSeed $randomSeed
              done
            }
            i=$((i+1))
          done
        done
      done
    done
  done
fi