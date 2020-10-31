import pandas as pd
import numpy as np
import math
import cv2
df =  pd.read_csv("300.3.ctf", sep='\s+')
phi1 = df['phi1'].to_numpy()
Phi = df['Phi'].to_numpy()
phi2 = df['phi2'].to_numpy ()

x = df['x'].to_numpy ()
y = df['y'].to_numpy ()

stepSize = 0.4
nRow = int(np.max(y) / stepSize) + 1
nCol = int(np.max(x) / stepSize) + 1

img = np.zeros((nRow, nCol, 3))
phi1 /= 360.0
Phi /= 360.0
phi2 /= 360.0

idxX = ((x + 0.001) / stepSize).astype(np.int32)
idxY = ((y + 0.001) / stepSize).astype(np.int32)

img[idxY, idxX, 0] = phi2
img[idxY, idxX, 1] = Phi
img[idxY, idxX, 2] = phi1

img = (img * 255).astype(np.uint8)

cv2.imwrite("300.3_selfparsed.png", img)