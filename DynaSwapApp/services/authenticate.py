import cv2
import numpy as np
import os
from sklearn.svm import SVC
from DynaSwapApp.services.face_utils import face_utils
from DynaSwapApp.services.face_models.MTCNN import MtcnnService
from DynaSwapApp.services.face_models.FNET import FnetService

class authenticate:
    def authenticate_image(self,image,rs_id):
        face_util = face_utils()
        # Preprocess
        try:
            image = face_util.align(image)
        except:
            raise ValueError("Multiple or no faces detected in image.")
            print("Multiple or no faces detected in image.")
        
        # Feature Extraction
        feature = face_util.extract(image)
        
        # Get RS Feature from database
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'database','rs_features.npy')
        rs_data = np.load(filename)

        rs_idx = rs_id-1
        rs_feature = rs_data[rs_idx,:]
        rs_feature = np.reshape(rs_feature[:-1],(1,512))

        # BioCapsule Generation
        bc = face_util.bc_fusion(-1.,feature,rs_feature)[0,:512].astype(float)
        return np.reshape(bc,(1,512))

    def authenticate_classifier(self,bc,classifier):
        prod = classifier.predict_proba(bc)
        if prod[0,0] >= prod[0,1]:
            return False, prod[0,1]
        return True, prod[0,1]