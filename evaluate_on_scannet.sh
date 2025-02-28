#!/bin/bash

MODE=$1
EXPNAME=$2

OUT_DIR=exps/goslam/scannet

scenes='0000'

echo "Start evaluating on ScanNet dataset..."

for sc in ${scenes};
do
  echo Running on $sc ...
  if [[ $MODE == "mono" ]]
  then
    python run.py configs/ScanNet/scene${sc}_mono.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME
  else
    python run.py configs/ScanNet/scene${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  fi
  echo $sc done!
done

echo Results for all scenes are:

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
done

echo All Done!
