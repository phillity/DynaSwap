import threading
import base64
import os
import sys
import io
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from django.http import JsonResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.views.generic import TemplateView
from django.core.files.base import ContentFile
from DynaSwapApp.models import Roles
from DynaSwapApp.models import Users
from DynaSwapApp.services.register import register
from DynaSwapApp.services.authenticate import authenticate
from django.utils import timezone

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html')

class RegisterPageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'register.html')

class AuthenticatePageView(TemplateView):
    def get(self, request, **kwargs):
        userName = request.GET.get('userName')
        roleId = request.GET.get('roleId')
        return render(request, 'authenticate.html', { 'userName' : userName, 'roleId' : roleId })

class AcceptedPageView(TemplateView):
    def get(self, request, **kwargs):
        userName = request.GET.get('userName')
        confidence = request.GET.get('confidence')
        roleId = request.GET.get('roleId')
        return render(request, 'accepted.html', { 'userName' : userName, 'confidence' : confidence, 'roleId' :  roleId})

class RejectedPageView(TemplateView):
    def get(self, request, **kwargs):
        userName = request.GET.get('userName')
        confidence = request.GET.get('confidence')
        roleId = request.GET.get('roleId')
        return render(request, 'rejected.html', { 'userName' : userName, 'confidence' : confidence, 'roleId' :  roleId})

class GetRolesView(TemplateView):
    def get(self, request, **kwargs):
        roles = []
        for role in Roles.objects.all().only('role_name', 'url'):
            roles.append({"role" : role.role_name, "url" : static(role.url)})
        return JsonResponse({"status": "true", "roles" : roles})

class RegisterView(TemplateView):
    __reg = register()

    def update_database(self,user_name,role_instance,bcs):
        # Train classifier using biocapsules
        classifier = self.__reg.register_classifier(bcs)

        # Convert biocapsules and classifier to binary to save in database
        classifier_binary = pickle.dumps(classifier)
        bcs_binary = pickle.dumps(bcs)

        # Save new user into database
        new_user = Users(user_name=user_name,role=role_instance,bio_capsule=bcs_binary,classifier=classifier_binary)
        new_user.save()
        print("register - database updated")
        return

    def post(self, request, **kwargs):
        ### Get POST data form
        try:
            # Extract data
            user_name = request.POST.get("userName", "")
            role = request.POST.get("role", "")
            images = []
            for key, value in request.POST.items():
                if key[:5] == "image":
                    format, imgstr = value.split(';base64,') 
                    ext = format.split('/')[-1] 
                    image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
                    images.append(image)
        except Exception as e:
            return JsonResponse({"status" : "false", "error" : "POST data error. " + str(e)})

        ### Check if username is already taken
        user_found = Users.objects.filter(user_name=user_name)
        if user_found.count() > 0:
            return JsonResponse({"status" : "taken", "error" : "Username " + user_name + " has already been registered."})
        
        ### Process POST data form
        try:
            # Get role info corresponding to chosen role
            role_instance = Roles.objects.filter(role_name=role)[0]
            role_id = role_instance.id

            # Convert submited images to biocapsules
            bcs = np.empty((0,513))
            for image in images:
                stream = image.file
                image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
                bc = self.__reg.register_image(image,role_id)
                bcs = np.vstack([bcs,bc])
        except Exception as e:
            return JsonResponse({"status" : "image", "error" : "Invalid input image. " + str(e)})

        ### Update database
        try:
            t = threading.Thread(target=self.update_database,args=(user_name,role_instance,bcs))
            t.setDaemon(True)
            t.start()
        except Exception as e:
            return JsonResponse({"status" : "error"})

        return JsonResponse({"status" : "true"})
        
class AuthenticateView(TemplateView):
    __reg = register()
    __auth = authenticate()

    def update_database(self,user_found,bc,bcs):
        # Update user bcs
        bc = np.reshape(np.append(bc,1.),(1,513))
        bcs = np.vstack([bcs,bc])
        bcs_binary = pickle.dumps(bcs)
        user_found.bio_capsule = bcs_binary

        # Update user classifier
        classifier = self.__reg.register_classifier(bcs)
        classifier_binary = pickle.dumps(classifier)
        user_found.classifier = classifier_binary

        # Save updated user into database
        user_found.last_authenticated = timezone.now()
        user_found.save()
        print("authenticate - database updated")
        return

    def post(self, request, **kwargs):
        ### Get POST data form
        try:
            # Extract data
            user_name = request.POST.get("userName", "")
            role = request.POST.get("role", "")
            temp_image = request.POST.get("image", "")
            format, imgstr = temp_image.split(';base64,') 
            ext = format.split('/')[-1] 
            image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)      
        except Exception as e:
            return JsonResponse({"status" : "false", "error" : "POST data error. " + str(e)})

        ### Check if username is already registered
        user_found = Users.objects.filter(user_name=user_name)
        if user_found.count() < 1:
            return JsonResponse({"status" : "not_taken", "error" : "Username " + user_name + " has not been registered."})
        user_found = user_found[0]

        ### Process POST data form
        try:
            # Get role info corresponding to chosen role
            role_instance = Roles.objects.filter(role_name=role)[0]
            role_id = role_instance.id
            
            # Convert submited images to biocapsules
            stream = image.file
            image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
            bc = self.__auth.authenticate_image(image,role_id)
        except Exception as e:
            return JsonResponse({"status" : "image", "error" : "Invalid input image. " + str(e)})

        ### Perform classification, Update database
        try:
            # Get bcs and classifier corresponding to username
            bcs = pickle.loads(user_found.bio_capsule)
            classifier = pickle.loads(user_found.classifier)

            # Perform authentication
            classification, prob = self.__auth.authenticate_classifier(bc,classifier)
            prob_string = str(np.around((prob * 100.),3)) + "%"

            # Classification outcome
            if classification == False:
                return JsonResponse({"status" : "false", "userName" : user_name, "confidence" : prob_string})

            # Update user with authentication time, new bc, new classifier
            if prob > 0.70:
                t = threading.Thread(target=self.update_database,args=(user_found,bc,bcs))
                t.setDaemon(True)
                t.start()

            # Update user with authentication time
            else:
                user_found.last_authenticated = timezone.now()
                user_found.save()
        except Exception as e:
            return JsonResponse({"status" : "error"})

        return JsonResponse({"status" : "true", "userName" : user_name, "confidence" : prob_string})