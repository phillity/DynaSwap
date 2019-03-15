import cv2
import numpy as np
import os
from sklearn.svm import SVC
from DynaSwapApp.services.face_utils import FaceUtils
from DynaSwapApp.services.face_models.MTCNN import MtcnnService
from DynaSwapApp.services.face_models.FNET import FnetService


class Register:
    def register_image(self, image, rs_id):
        face_util = FaceUtils()
        # Preprocess
        try:
            image = face_util.align(image)
        except:
            raise ValueError('Multiple or no faces detected in image.')
            print('Multiple or no faces detected in image.')

        # Feature Extraction
        feature = face_util.extract(image)
        feature_flip = face_util.extract(cv2.flip(image, 1))

        # Get RS Feature from database
        curr_path = os.path.dirname(__file__)
        filename = os.path.join(curr_path, 'data', 'rs_features.npy')
        rs_data = np.load(filename)['arr_0']

        rs_idx = rs_id - 1
        rs_feature = rs_data[rs_idx, :]
        rs_feature = np.reshape(rs_feature[:-1], (1, 512))

        # BioCapsule Generation
        bc = face_util.bc_fusion(1., feature, rs_feature)
        bc_flip = face_util.bc_fusion(1., feature_flip, rs_feature)
        bcs = np.vstack([bc, bc_flip])

        # Get BioCaspules for other RSs
        rs_ids = np.arange(0, 4, 1)
        rs_ids = np.delete(rs_ids, (rs_idx))
        for rs_idx in rs_ids:
            rs_feature = rs_data[rs_idx, :]
            rs_feature = np.reshape(rs_feature[:-1], (1, 512))
            bc = face_util.bc_fusion(0., feature, rs_feature)
            bc_flip = face_util.bc_fusion(0., feature_flip, rs_feature)
            bcs = np.vstack([bcs, bc])
            bcs = np.vstack([bcs, bc_flip])
        return bcs

    def register_classifier(self, bcs):
        # Load dummy features to use as negative examples
        curr_path = os.path.dirname(__file__)
        filename = os.path.join(curr_path, 'database', 'dummy_bc.npy')
        dummy = np.load(filename)

        train = np.vstack([bcs, dummy])
        y = train[:, -1]
        train = train[:, :-1]

        classifier = SVC(kernel='rbf', C=1.0, degree=3, gamma='auto', probability=True)
        classifier.fit(train, y)

        return classifier
