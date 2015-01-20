import cv2
import numpy as np
import os 
import sys
from copy import copy
# Load the images

formats = ['jpg','png','tif','jpeg']                                            #Allowed formats

def knn(img1,images):
    # img =cv2.imread("main.jpg")
    img = cv2.imread(img1)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    surf = cv2.SURF(128)                                                        # SURF keypoint extraction
    kp = surf.detect(imgg)
    kp, descritors = surf.compute(imgg,kp)
    # surf = cv2.SURF(1024)

    samples = np.array(descritors)                                              # Setting up samples and responses for kNN
    responses = np.arange(len(kp),dtype = np.float32)

    knn = cv2.KNearest()                                                          # kNN training
    knn.train(samples,responses)

    # modelImages = ["main1.jpg", "main2.jpg","man1.png"]#, "grup4.jpg"]
    modelImages = images[:]

    for modelImage in modelImages:

        template = cv2.imread(modelImage)                                       # loading a template image and searching for similar keypoints
        templateg= cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        keys = surf.detect(templateg)

        keys,desc = surf.compute(templateg, keys)
        count = 0
        lent = len(desc)

        for h,des in enumerate(desc):
            # print h#,des                                                        #debug 
            # print des.shape
            des = des.astype(np.float32).reshape((-1,128))

            retval, results, neigh_resp, dists = knn.find_nearest(des,1)
            res,dist =  int(results[0][0]),dists[0][0]


            if dist<0.1:                                                        # draw matched keypoints in red color
                # color = (0,0,255)
                count += 1 
            sim = count*100/lent    
            # else:                                                                   # draw unmatched in blue color
                #print dist
                # color = (255,0,0)

            # x,y = kp[res].pt                                                      #Draw matched key points on original image
            # center = (int(x),int(y))
            # cv2.circle(img,center,2,color,-1)
            # x,y = keys[h].pt                                                    #Draw matched key points on template image
            # center = (int(x),int(y))
            # cv2.circle(template,center,2,color,-1)
            
        print img1,modelImage,sim
        if sim < 80:
            # print img1,modelImage
            del images[images.index(modelImage)]
        # else:                                                                     #debug
            # print img1,modelImage,"sim =",sim
        # print "simmilarity =",((count)*100/lent)
        # cv2.imshow('img',img)                                                     #plot features
        # cv2.imshow('tm',template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return images


if __name__ ==  "__main__":
    files = []
    for fil in os.listdir(sys.argv[1]):
        if os.path.isfile(os.path.join(sys.argv[1],fil)) and fil.split('.')[-1].lower() in formats:
            files.append(os.path.join(sys.argv[1],fil))
    files.sort()
    while files:
        img1 = files[0]                                                         #get first image from list 
        # print files
        del files[0]                                                             #remove first image so it doesnt compare to itself
        imgs = knn(img1,files[:])                                                 #get list of similar images
        # print ','.join(imgs)," are similar to ",img1
        for i in imgs:                                                            #remove all similar images from list and delete them from directory
            files.remove(i)
            os.remove(i)
        # files = [i for i in files if i not in imgs]
        # print files

            