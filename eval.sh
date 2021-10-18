if [ ! $1 ];
then
    DATA_DIR='/home/zhicai/ImageNet2012/val'
else
    DATA_DIR="$1"
fi
if [ ! $2 ];
then
    MODEL_DIR='/data/zhicai/ckpts/Mgmlp/train/20211005-203106-nest_gmlp_s-224'
else
    MODEL_DIR="$2"
fi
python3 validate.py $DATA_DIR  --model nest_gmlp_s --checkpoint $MODEL_DIR/model_best.pth.tar --no-test-pool --amp --img-size 224 -b 16

