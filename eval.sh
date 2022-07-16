if [ ! $1 ];
then
    DATA_DIR='/scratch/zhicai/ImageNet2012/val'
else
    DATA_DIR="$1"
fi
if [ ! $2 ];
then
    MODEL_DIR='/data1/zhicai/ckpts/Mgmlp/train/20220701-120252-Create_PosMLP-224'
else
    MODEL_DIR="$2"
fi
python3 validate.py $DATA_DIR  --model Create_PosMLP --checkpoint $MODEL_DIR/model_best.pth.tar --no-test-pool --amp --img-size 224 -b 64

