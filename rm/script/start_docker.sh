
docker run -it  --ulimit memlock=-1   --net host --oom-kill-disable -d -v /chubao/bella:/data_belle -v /nfs/a100-80G-20:/model_save20  -v /nfs/a100-80G-19:/model_save19    -v  /data/nfs14/nfs/aisearch/asr/dengyong013:/code  --shm-size="16g"  -v  /nfs/v100-022/:/data  --name  pythondy   --runtime=nvidia  --restart=always --privileged  bestpredicts/bestpredict:base4.8
