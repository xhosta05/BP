from scipy.sparse import spdiags,csc_matrix,linalg 
from sklearn import preprocessing

import pandas as pd
import numpy as np
from numpy import loadtxt

import os 
import re 

def low_pass_filter(adata: np.ndarray, bandlimit: int = 10, sampling_rate: int = 100) -> np.ndarray:
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)    
    fsig = np.fft.fft(adata)        
    for i in range(bandlimit_index + 1, len(fsig) - bandlimit_index ):
        fsig[i] = 0            
    adata_filtered = np.fft.ifft(fsig)    
    return np.real(adata_filtered)

def one_hot_encode(arr,classes):
    ohe= preprocessing.OneHotEncoder(sparse=False,categories=classes)
    transformed = ohe.fit_transform(arr)
    # transformed=(transformed.toarray())
    
    return transformed
    
def data_normalize(arr):
    St_Idx = 0
    End_Idx = 2047

    DataLoaded = arr[:, St_Idx:End_Idx]
    Rows, Cols = (DataLoaded.shape)
    Dataset_Normalized = np.zeros([Rows, Cols])

    for Row_Idx in range(Rows):
        Max_Value = DataLoaded[Row_Idx, :].max()
        Dataset_Normalized[Row_Idx] = DataLoaded[Row_Idx, :]/Max_Value
    return Dataset_Normalized


def baseline_als(y, lam=1e6, p=0.5, niter=10):
    L = len(y)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):        
        
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def subtract_baseline(arr, lam=1e6, p=0.5, niter=10):
	Database_baseline_subtracted= np.zeros(arr.shape)
	for i,meas in enumerate(arr):
	  baseline=baseline_als(meas)
	  res = np.subtract(meas, baseline)
	  Database_baseline_subtracted[i]=res
	  
	return Database_baseline_subtracted

def text_read(path): # todo multiple lines
  with open(path, 'r', encoding="utf-8", errors='replace') as file:
      for row in file:
        arr=row.split(' ')
        arr = np.array(arr[:-1]).astype(int)
  return arr

def txt2csv(path, shape):
  pic=np.empty(shape=shape)
  for root, dirs, files in os.walk(path, topdown=False):
      for name in files:
        if re.findall('_\d+_\d+.txt', name)  : #todo:there's gotta be a better way to do this
          
            arr=[]
            idxs=re.findall(r'\d+', name)
            arr= text_read(os.path.join(root, name))

            pic[int(idxs[0])][int(idxs[1])] = arr
#   new_pic = pic.reshape(-1, pic.shape[-1])  # flatten 128x128 to 1 dimension
  return pic #, new_pic

def csv2data(pic_data_path):
    pixels = loadtxt(pic_data_path, delimiter=',')
    Dataset_Normalized = data_normalize(pixels)
    return Dataset_Normalized
    
def arr2pic(arr, shape, model):
    Max_XScanIdx = shape[0]
    Max_YScanIdx = shape[1]
    RamanImage = np.zeros((Max_XScanIdx, Max_YScanIdx))
    
    Model_Prediction = model.predict(arr)  
    for Idx_X in range(Max_XScanIdx):
        for Idx_Y in range(Max_YScanIdx):
            idx=Idx_X * Max_YScanIdx +  Idx_Y
            if len(Model_Prediction)>idx:
                RamanImage[Idx_X][Idx_Y]= np.argmax(Model_Prediction[idx])
    return  RamanImage
 
    
def data2df(dataset):
	"""
	Converts array with raman measurments into pandas dataframe.

	@param dataset: array with class in the first column and 1 raman measurment per row 
	@return: pandas dataframe  
	"""
	rows_idx  = [str(i) for i in range(len(np.squeeze(dataset[:, :1])))]
	df_concat = np.concatenate((np.reshape([rows_idx], (-1,1)) , dataset), axis=1)

	cols_idx  = [str(i) for i in range(len(np.squeeze(dataset[:1, :] ))-1)]
	cols_idx = (["classes"]+cols_idx)
	cols_idx = (["rows_idx"]+cols_idx)
	df_concat = np.concatenate(([cols_idx], df_concat) )

	df_train = pd.DataFrame(data=df_concat[1:,1:],    # values
		                    index=df_concat[1:,0],    # 1st column as index
		                    columns=df_concat[0,1:])  # 1st row as the column names
	return df_train

def path2pic(pic_data_path, shape, model):
    pixels=csv2data(pic_data_path)
    # print(pixels.shape)

    pic=arr2pic(pixels,shape,model)
    return pic

def save_arr(arr,path):
	np.savetxt(path, arr, delimiter=",")
    
