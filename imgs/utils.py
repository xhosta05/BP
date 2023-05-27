import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import rescale_intensity

def get_gray(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
    
def label_regions(binary_image, kernel_size):
    # <input> binary_image: an image containing foreground in white and background in black
    # <return>: labeled_image: a single-channel image, where each connected set in the clean version of "binary_image" is assigned to a numeric label (with background = 0)
    
    image = np.array(binary_image, dtype=np.uint8)
    
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                   
    return cv2.connectedComponentsWithStats(image) # (labelsCount, labelIds, values, centroid)
    
def segment_image(image, treshold):
    image=get_gray(image)
    particles = np.invert(image > treshold) 
    return particles
    
def getMask(image,threshold=100,kernelSize=7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    particles_mask = segment_image(image,threshold)  # img2binaryImg
    # todo check other imgs (auto threshold)
    labelsCount, labelIds, values, centroid=label_regions(particles_mask, kernelSize) #image morphology
    resultMask = (labelIds == 0).astype("uint8") * 255
	# output = cv2.bitwise_not(componentMask)
    return resultMask

def contour2img(image,contour,savePath=None):    
    img = image.copy() # Draw the contour into image
    
    for (x, y) in contour: # iterate through pixels
        cv2.circle(img, (x, y), 1, (255, 0, 0), 6)
    if savePath is not None:
        cv2.imwrite(savePath, img)
    return img

def markContours(imageMask,savePath=None):
    contours = cv2.findContours(imageMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # img, contours, hierarchy
    contoutrsList=[]
    for i,cnt in enumerate(contours[0]): # iterate through contours
        if i > 0:  # skip background
            contour = cnt.squeeze()
            if savePath is not None:
                img=contour2img(imageMask,contour,savePath=savePath)   
            else:
                img=contour2img(imageMask,contour)   
                contoutrsList.append(img)
                    
    return contoutrsList

def img2segmented(imageMask,save=""):
    contours = cv2.findContours(imageMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # img, contours, hierarchy
    image0 = np.zeros(imageMask.shape, dtype=np.uint8)
    contoursImg = np.zeros(imageMask.shape, dtype=np.uint8)
    for i,cnt in enumerate(contours[0]): # iterate through contours
        if i > 0:  # skip background
            # todo extract features
            # todo save features + contour path to COCO-like dataset 
            contoursImg = cv2.drawContours(contoursImg, contours[0],i,i, thickness=cv2.FILLED)
#             contoursGreyImg=contoursImg[:,:,:1] #      
            if save != "":
                cv2.imwrite(save, contoursImg)
    return contoursImg

def imgPath2masks(img_paths,threshold=110,kernelSize=5,savepathSlices=None, savepathContours=None):
    for imgPath in img_paths:
        image = cv2.imread(imgPath)
        imageMask = utils.getMask(image,threshold=threshold,kernelSize=kernelSize) 

        if savepathSlices is not None:
            utils.img2segmented(imageMask,savepathSlices)
        if savepathContours is not None:
            markContours(imageMask,save=savepathContours)    

def distanceMap(mask_slices):
	# TODO make friendly for individual particle slices instead binary image
	# TODO make particles weights 1 and outlines higher (1-10), convert to float ofc
    outlines = np.zeros(mask_slices.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_slices.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlines, contours, -1, 1, thickness=cv2.FILLED)
#         cv2.drawContours(outlines, contours, -1, 1, thickness=2)
    outlines = cv2.normalize(outlines, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    dist = cv2.distanceTransform(cv2.bitwise_not(outlines), cv2.DIST_LABEL_PIXEL, cv2.DIST_MASK_PRECISE)
    normalized = cv2.normalize(dist, -1, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).astype(np.uint8)
    normalized = rescale_intensity(normalized, in_range=(0,20), out_range=(0,255)).astype(np.uint8)
    normalized = cv2.bitwise_not(normalized)
    
    res = cv2.bitwise_or(normalized,outlines)
    return res
    
def extractFeaturesFromContour(contour):
    area = cv2.contourArea(contour)
    cntLen = cv2.arcLength(contour, closed=True)
    _,radius = cv2.minEnclosingCircle(contour)
    _,_,wBB,hBB = cv2.boundingRect(contour)
    k = cv2.isContourConvex(contour)

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    rectMin = cv2.minAreaRect(contour) 

    rectMinRatio = min(rectMin[1])/max(rectMin[1]) # height & width of min rectangle
    extent = float(area)/ (wBB*hBB)  # cnt area / BBox area
    solidity = float(area)/hull_area # cnt area / k hull area
    equi_diameter = np.sqrt(4*area/np.pi)
    
#     moments = cv2.moments(contour)
#     bigSqrt = sqrt((moments["m20"]-moments["m02"]) * (moments["m20"]-moments["m02"].) +4*moments["m11"]*moments["m11"])
#     eccentricity = double((moments["m02"]+bigSqrt) / (moments["m20"]+moments["m02"]-bigSqrt))
#     mean_val = cv.mean(im,mask = mask) # mean color/intensity

    features=np.array([rectMinRatio, extent, solidity, equi_diameter, area, cntLen, radius, k],dtype=float)
    return features
########################   Visualizing    ###########################################

def plot_intensity_histogram(image):
    # <input> image: a color (3 channels) image
    # <output>: plot with the grayscale intensity histogram
    if len(image.shape)==3:
    	gray=get_gray(image)
    else:
    	gray=image    	
    
    histogram, bin_edges = np.histogram(gray, bins=256)
    
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(bin_edges[0:-1], histogram)  #
    plt.show()
    
