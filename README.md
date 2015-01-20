Image-similarity
===============

Different programs for calculating percentage similarity of two images

*	hashes.py - Calculate Image similarity based on phash
*	drawMatches.py - mapping of similar points of 2 images
*	knn.py - iterate through all images in given folder to find similar images (sim % > 80) and delete them 
*	knnsingle.py - mapping similar points in 2 images using knn

Usage:
*	python < hashes.py / drawMatches.py / knnsingle.py > < img1 > < img2 >
*	python knn.py < dir >
