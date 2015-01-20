import cv2
import numpy as np
import sys
# Load the images

if sys.argv > 2:
    img =cv2.imread(sys.argv[1])

    # Convert them to grayscale
    imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # imgg = cv2.fastNlMeansDenoising(imgg,None,10,10,20)

    # SURF extraction
    surf = cv2.SURF(128)
    kp = surf.detect(imgg)
    kp, descritors = surf.compute(imgg,kp)
    # surf = cv2.SURF(1024)

    # Setting up samples and responses for kNN
    samples = np.array(descritors)
    responses = np.arange(len(kp),dtype = np.float32)

    # kNN training
    knn = cv2.KNearest()
    knn.train(samples,responses)

    modelImages = sys.argv[2:]

    for modelImage in modelImages:

        template = cv2.imread(modelImage)                                       #loading a template image and searching for similar keypoints
        templateg= cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        # templateg = cv2.fastNlMeansDenoising(templateg,10,10,10,20)
        keys = surf.detect(templateg)

        keys,desc = surf.compute(templateg, keys)                               # get surf keypoints from template image
        count = 0
        lent = len(desc)
        for h,des in enumerate(desc):
            # print h#,des
            # print des.shape
            des = des.astype(np.float32).reshape((-1,128))

            retval, results, neigh_resp, dists = knn.find_nearest(des,1)         
            res,dist =  int(results[0][0]),dists[0][0]


            if dist<0.1:                                                        # draw matched keypoints in red color
                color = (0,0,255)
                count += 1

            else:                                                               # draw unmatched in blue color
                #print dist
                color = (255,0,0)

            x,y = kp[res].pt                                                    # draw matched key points on original image
            center = (int(x),int(y))
            cv2.circle(img,center,2,color,-1)

            x,y = keys[h].pt                                                    # draw matched key points on template image
            center = (int(x),int(y))
            cv2.circle(template,center,2,color,-1)
     
            # print ""

        cv2.imshow('img similarity = %s'%(count*100/lent),img)                  #similarity = TP/P = count/lent
        cv2.imshow('tm',template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()