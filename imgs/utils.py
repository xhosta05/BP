import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    
