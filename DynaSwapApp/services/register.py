import cv2
import numpy as np
import sys
import os
from sklearn.svm import SVC
from DynaSwapApp.services.face_utils import face_utils
from DynaSwapApp.services.face_models.MTCNN import MtcnnService
from DynaSwapApp.services.face_models.FNET import FnetService

class register:
    def register_image(self,user_id,image,rs_id):
        face_util = face_utils()
        # Preprocess
        try:
            image = face_util.align(image)
        except:
            raise ValueError("Multiple or no faces detected in image.")
            print("Multiple or no faces detected in image.")
        
        # Feature Extraction
        feature = face_util.extract(image)
        feature_flip = face_util.extract(cv2.flip(image,1))
        
        # Get RS Feature from database
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'database','rs_features.npy')
        rs_data = np.load(filename)

        rs_idx = rs_id-1
        rs_feature = rs_data[rs_idx,:]
        rs_feature = np.reshape(rs_feature[:-1],(1,512))

        # BioCapsule Generation
        bc = face_util.bc_fusion(user_id,feature,rs_feature)
        bc_flip = face_util.bc_fusion(user_id,feature_flip,rs_feature)
        bcs = np.vstack([bc,bc_flip])

        # Get BioCaspules for other RSs
        rs_ids = np.arange(0,4,1)
        rs_ids = np.delete(rs_ids,(rs_idx))
        for rs_idx in rs_ids:
            rs_feature = rs_data[rs_idx,:]
            rs_feature = np.reshape(rs_feature[:-1],(1,512))
            bc = face_util.bc_fusion(0.,feature,rs_feature)
            bc_flip = face_util.bc_fusion(0.,feature_flip,rs_feature)
            bcs = np.vstack([bcs,bc])
            bcs = np.vstack([bcs,bc_flip])

        return bcs

    def register_classifier(self,user_id,bcs):
        # Load dummy features to use as negative examples
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'database','dummy_bc.npy')
        dummy = np.load(filename)

        idx = np.where(bcs[:,-1] == user_id)
        bcs[idx,-1] = 1.
        bcs = bcs.astype(float)

        train = np.vstack([bcs,dummy])
        y = train[:,-1]
        train = train[:,:-1]

        classifier = SVC(kernel='rbf',C=1.0,degree=3,gamma='auto')
        classifier.fit(train,y)

        return classifier
        

