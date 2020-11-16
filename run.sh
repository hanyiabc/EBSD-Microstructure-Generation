python.exe ./src/TextureSynthesis.py -s ./data/small/*.png -o ./data/result/ -n 25 -sw 1.0 -tw 0.05 -hw 1e10

python.exe ./src/segmentation.py -i ./data/result/*.png -o ./data/result/cleaned/ -d ./data/small/

python.exe ./src/segmentation.py -i ./data/result/hr/*.png -o ./data/result/hr/cleaned/ -d ./data/small/