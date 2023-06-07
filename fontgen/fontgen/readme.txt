 conda activate dynew1
  654  cd dy/DG-Font-main/DG-Font-main/
  655  python main.py --gpu 0 --data_path font/png/dy77
  656  python font2img.py --ttf_path font/dy77 --chara 1077.txt --save_path font/png/lx --img_size 200 --chara_size 200
  657  python font2img.py --ttf_path font/dy77 --chara 1077.txt --save_path font/png/dy77 --img_size 200 --chara_size 200
  658  python main.py --gpu 0 --data_path font/png/dy77
  659  python font2img.py --ttf_path font/dy77 --chara 1077.txt --save_path font/png/dy77 --img_size 200 --chara_size 200
  660  python main.py --gpu 0 --data_path font/png/dy77
  661  python font2img.py --ttf_path font/newziti --chara 1077.txt --save_path font/png/newziti --img_size 200 --chara_size 200
  662  python main.py --gpu 0 --data_path font/png/newziti
  663  python font2img.py --ttf_path font/lyj --chara 1077.txt --save_path font/png/lyj --img_size 200 --chara_size 200
  664  python font2img.py --ttf_path font/lyj --chara lyj.txt --save_path font/png/lyj --img_size 200 --chara_size 200
  665  python main.py --gpu 0 --data_path font/png/lyj
  666  python main.py --gpu 0 --data_path font/png/newziti

