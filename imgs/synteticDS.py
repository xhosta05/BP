import matplotlib.pyplot as plt
import time
import os
import re
import cv2

# load resnet
import numpy as np
import torchvision
from torchvision import datasets, models, transforms # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html    

# train
import torch
import torch.nn as nn
import torch.optim as optim

# generate img
import copy    
import random
import itertools
from imutils import rotate_bound
from random import randint, shuffle

import utils
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size


def backgroundFrameMean(imagePaths,ratio=False):
    shape= cv2.imread(imagePaths[0]).shape
    background = np.zeros(shape, dtype=np.float32)
    imgCount=len(imagePaths)
    
    if not ratio:
    	ratio=imgCount
    
    for idx,imgPath in enumerate(imagePaths): #skip 1st image
#         print(i)
        currentImg= cv2.imread(imgPath)
        cv2.accumulateWeighted(currentImg,background,1/ratio) # src,dst,ratio(Weight of the input image)
    return background.astype(np.uint8)

def getPaths(data_path,leafsOnly=True):
    img_paths=[]
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if re.findall('.jpg||.png', name):
                img_paths.append(os.path.join(root,name))
        if leafsOnly:
        	break
    if not img_paths:
        print("err loading ",img_paths)
    else:
        return img_paths
        
def cutoutsFromImgs(img_paths):
# at the moment, it saves the cutouts into  'particleCutouts' dir in img_paths
#     todo change^
    for imgPath in img_paths:    

        image = cv2.imread(imgPath)
        imageMask = utils.getMask(image,threshold=120,kernelSize=5) 
        contours = cv2.findContours(imageMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # img, contours, hierarchy

        labelsCount, _, values, _=  cv2.connectedComponentsWithStats(imageMask) 
        contoursImg = np.zeros(image.shape, dtype=np.uint8)
        axSize = math.sqrt(labelsCount)

        try:
            for i,cnt in enumerate(contours[0]): # iterate through contours
                if i > 0 and cv2.contourArea(cnt.squeeze())>30:  # skip background and dust
                    contour = cnt.squeeze()
                    xmin,ymin,w,h = cv2.boundingRect(contour)  #  xmin(left),ymin(top),w,h

                    cutoutOrig = (image[ymin:ymin+h,xmin:xmin+w])
                    cutoutMask =imageMask[ymin:ymin+h,xmin:xmin+w]
                    cutoutMask =cv2.cvtColor(cutoutMask,cv2.COLOR_GRAY2RGB)

                    cutout = cv2.bitwise_and(cutoutOrig,cutoutMask)

                    crackPth= os.path.join(save_data_path,"particleCutouts")
                    filename, file_extension = os.path.splitext(os.path.basename(imgPath))
                    savePath= os.path.join(crackPth, filename+"_"+str(i)+file_extension)
                    cv2.imwrite(savePath, cutout)
        except Exception as e:
            print(crackPth)
            print(e)
    
def sortImgs2dirs(classResults,imageFolder,save2Path=None):
    # predict function -> sort dataset
    # list/arr classResults(1d) containing classes for images in imageFolder
    # datasets.ImageFolder object
    for idx, result in enumerate(resArg):
        label=imageFolder.imgs[idx][1] # ImageFolder.imgs ->tuples  (imagePath, class)
        
        src = imageFolder.imgs[idx][0]    
        if not save2Path:
            save2Path=os.path.dirname(src)
            
        #     TODO checkif dst dir exists, make if not
        dst= os.path.join(save2Path,str(label))
        dst= os.path.join(dst,os.path.basename(imageFolder.imgs[idx][0]))
        os.replace(src, dst)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device="cpu", is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def Data_transforms(input_size):
	 return {
		'train': transforms.Compose([
		transforms.RandomResizedCrop(input_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
