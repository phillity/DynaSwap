import base64
import os
import io
import time
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.forms import forms
from django.http import JsonResponse
from django.http.response import HttpResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.views.generic import TemplateView
from django.core.files.base import ContentFile
from DynaSwapApp.models import Roles
from DynaSwapApp.models import Users
from DynaSwapApp.services.register import register
from DynaSwapApp.services.face_models import MTCNN
from DynaSwapApp.services.face_models import FNET

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html')

class RegistrationView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'register.html')

class AuthenticationView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'authenticate.html')

class GetRolesView(TemplateView):
    def get(self, request, **kwargs):
        roles = []
        for role in Roles.objects.all().only('role_name', 'url'):
            roles.append({"role" : role.role_name, "url" : static(role.url)})
        return JsonResponse({"status": "true", "roles" : roles})

class UploadImageView(TemplateView):
    def post(self, request, **kwargs):
        # Get POSTed data form
        try:
            userName = request.POST.get("userName", "")
            #usersFound = Users.objects.filter(user_name=userName)
            #if userFound.
                #return JsonResponse({"status" : "show", "error" : "User " + userName + " has already been defined"})

            images = []
            for key, value in request.POST.items():
                if key[:5] == "image":
                    format, imgstr = value.split(';base64,') 
                    ext = format.split('/')[-1] 
                    image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
                    images.append(image)

            role = request.POST.get("role", "")
        except Exception as e:
            return JsonResponse({"status" : "false", "error" : "POST data error. " + str(e)})
        
        # Process POSTed data form
        mtcnn_model = MTCNN.MTCNN()
        facenet_model = FNET.FNET()
        reg = register(mtcnn_model,facenet_model)
        try:
            # Convert submited images to bcs
            bcs = np.empty((0,513))
            for image in images:
                stream = image.file
                image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
                bc = reg.register_image(userName,image,role)
                bcs = np.vstack([bcs,bc])

            # Train classifier using bcs
            classifier = reg.register_classifier(userName,bcs)

            classifier_binary = pickle.dumps(classifier)
            bcs_binary = pickle.dumps(bcs)

            role_instance = Roles.objects.filter(role_name=role)[0]
            new_user = Users(user_name=userName,role=role_instance,bio_capsule=bcs_binary,classifier=classifier_binary)
            new_user.save()
        except Exception as e:
            return JsonResponse({"status" : "false", "error" : "Process data error. " + str(e)})

        return JsonResponse({"status":"true"})
        
