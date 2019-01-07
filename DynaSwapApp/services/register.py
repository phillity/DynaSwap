import cv2
import numpy as np
import sys
from sklearn.svm import SVC
from DynaSwapApp.services.face_utils import face_utils
from DynaSwapApp.services.face_models import MTCNN
from DynaSwapApp.services.face_models import FNET

class register:
    def __init__(self,mtcnn_model,facenet_model):
        # Create face_utils object with FaceNet and MTCNN models
        self.__face = face_utils(mtcnn_model,facenet_model)

    def register_image(self,user_id,image,rs_id):
        # Preprocess
        try:
            image = self.__face.align(image)
        except:
            raise ValueError("Bad Input Image")
            print("Bad Input Image")
        
        # Feature Extraction
        feature = self.__face.extract(image)
        feature_flip = self.__face.extract(cv2.flip(image,1))
        
        # Get RS Feature from database
        rs_idx = -1
        if rs_id == "President":
            rs_idx = 0
        elif rs_id == "CEO":
            rs_idx = 1
        elif rs_id == "Manager":
            rs_idx = 2
        else:
            rs_idx = 3
        rs_data = np.load("DynaSwapApp//services//database//rs//rs_features.npy")

        rs_feature = rs_data[rs_idx,:]
        rs_feature = np.reshape(rs_feature[:-1],(1,512))

        # BioCapsule Generation
        bc = self.__face.bc_fusion(user_id,feature,rs_feature)
        bc_flip = self.__face.bc_fusion(user_id,feature_flip,rs_feature)
        bcs = np.vstack([bc,bc_flip])

        # Get BioCaspules for other RSs
        rs_ids = np.arange(0,4,1)
        rs_ids = np.delete(rs_ids,(rs_idx))
        for rs_idx in rs_ids:
            rs_feature = rs_data[rs_idx,:]
            rs_feature = np.reshape(rs_feature[:-1],(1,512))
            bc = self.__face.bc_fusion(0.,feature,rs_feature)
            bc_flip = self.__face.bc_fusion(0.,feature_flip,rs_feature)
            bcs = np.vstack([bcs,bc])
            bcs = np.vstack([bcs,bc_flip])

        return bcs

    def register_classifier(self,user_id,bcs):
        # Load dummy features to use as negative examples
        dummy = np.load("DynaSwapApp//services//database//dummy//dummy_bc.npy")

        idx = np.where(bcs[:,-1] == user_id)
        bcs[idx,-1] = 1.
        bcs = bcs.astype(float)

        train = np.vstack([bcs,dummy])
        y = train[:,-1]
        train = train[:,:-1]

        classifier = SVC()
        classifier.fit(train,y)

        return classifier
        

