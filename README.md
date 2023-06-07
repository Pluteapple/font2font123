# font2font123

数据集均是汉仪所下字库，由font2img转成字体图像


<img width="655" alt="image" src="https://github.com/apple951/font2font123/assets/53884371/1e368688-06a9-494c-bac7-98163a750213">


## font2font操作

1  --dst_font 目标字体   执行命令前 先删除test内的图片 
 
python font2img.py --src_font ./font/yuan.ttf  --dst_font ./font/AliHYAiHei.ttf  --charset=CN --sample_count=200 --sample_dir=./experiment/testimg  --label=0 --filter=1 --shuffle=0 --char_size=220

2 python package.py --dir ./experiment/testimg --save_dir ./experiment/data --split_ratio 0.1

3 python train.py --experiment_dir ./experiment --experiment_id=73 --batch_size=8  --lr=0.002 --epoch=400 --sample_steps=40 --schedule=20 --L1_penalty=100 --Lconst_penalty=15

## fontgen操作


conda activate lxnew1

cd Font-main/Font-main/

python main.py --gpu 0 --data_path font/png/lx77

python font2img.py --ttf_path font/lx77 --chara 1077.txt --save_path font/png/lx --img_size 200 --chara_size 200

python main.py --gpu 0 --data_path font/png/lx77

## transformer操作

$ python test.py --content YOUR IMAGE PATH --style YOUR IMAGE PATH --output PATH

### You can train a new model following these steps:

1. Download the fontdata
2. $ python train.py 


