import torchvision.transforms as T
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader


class DataSetCustom(torch.utils.data.Dataset):
    def __init__(self, inImgs, gtImgs, augImgs=[], transformIn=None, transformGT=None):
        self.inImgs=inImgs
        self.gtImgs=gtImgs
        self.augImgs=augImgs
        self.transformIn = transformIn
        self.transformGT = transformGT

    def __len__(self):
        return len(self.inImgs)

    def __getitem__(self, idx):
        # inImg = read_image(self.inImgs[idx])
        # inImg = self.inImgs[idx]
        # gtImg = self.gtImgs[idx]

        inImg = cv2.imread(self.inImgs[idx])
        
        gtImg = cv2.imread(self.gtImgs[idx])
        threshold, r = cv2.threshold(gtImg[:,:,0], 100, 255, cv2.THRESH_BINARY)
        threshold, g = cv2.threshold(gtImg[:,:,1], 100, 255, cv2.THRESH_BINARY)
        threshold, b = cv2.threshold(gtImg[:,:,2], 100, 255, cv2.THRESH_BINARY)
        gtImg=np.stack((r,g,b),axis=2)
        
        if self.augImgs:
            augImg = cv2.imread(self.augImgs[idx])
        else:
            augImg = np.array([])

        if self.transformIn:
            inImg = self.transformIn(inImg)
        if self.transformGT:
            gtImg = self.transformGT(gtImg)
        if self.transformIn and np.any(augImg):
            augImg = self.transformIn(augImg)
            
        return inImg, gtImg, augImg
        
