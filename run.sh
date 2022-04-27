python.exe ./src/TextureSynthesis.py -s ./data/small/sim3_full.png -o ./data/result/ -n 1000 -sw 1e-6 -tw 1e-8
python.exe ./src/TextureSynthesistf2.py -s ./data/small/sim3_full.png -o ./data/result/ -n 1000 -sw 1e-6 -tw 1e-8


python.exe ./src/segmentation.py -i ./data/result/*.png -o ./data/result/cleaned/ -d ./data/small/

python.exe ./src/segmentation.py -i ./data/result/hr/*.png -o ./data/result/hr/cleaned/ -d ./data/small/