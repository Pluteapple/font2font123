# font2font123
#1  --dst_font 目标字体   执行命令前 先删除test内的图片 
 
#python font2img.py --src_font ./font/yuan.ttf  --dst_font ./font/AliHYAiHei.ttf  --charset=CN --sample_count=200 --sample_dir=./experiment/testimg  --#label=0 --filter=1 --shuffle=0 --char_size=220

#2 python package.py --dir ./experiment/testimg --save_dir ./experiment/data --split_ratio 0.1

#3python train.py --experiment_dir ./experiment --experiment_id=73 --batch_size=8  --lr=0.002 --epoch=400 --sample_steps=40 --schedule=20 --L1_penalty=100 --Lconst_penalty=15
