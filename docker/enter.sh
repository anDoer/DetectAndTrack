if test "$#" -lt 1; then
  echo "No GPU specified! Please pass gpu ids, i.e. 0,1,2"
  echo "I will grant the docker container access to all gpus now"
  gpus='all'
else
  gpus='"device='"$1"'"'
fi

./build.sh 

DATASET_DIR="/media/datasets/"
WORK_DIR="/media/work2/"
DATA_DIR="/media/data/"
XSERVER="/tmp/.X11-unix/"

docker run\
     --gpus $gpus\
    --shm-size="20g"\
    -v "$DATASET_DIR":/media/datasets\
    -v "$WORK_DIR":/media/work2\
    -v "$XSERVER":/tmp/.X11-unix\
    -v "$DATA_DIR":/media/data\
    -e DISPLAY=unix$DISPLAY\
    --rm -it\
    detectandtrack \
    /bin/bash
