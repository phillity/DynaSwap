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
from DynaSwapApp.models import UsersRoles
from DynaSwapApp.models import DynaSwapUsers
from DynaSwapApp.services.register import Register
from DynaSwapApp.services.authenticate import Authenticate
from django.utils import timezone


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
        return render(request, 'authenticate.html', {'userName': userName, 'roleId': roleId})


class AcceptedPageView(TemplateView):
    def get(self, request, **kwargs):
        userName = request.GET.get('userName')
        confidence = request.GET.get('confidence')
        roleId = request.GET.get('roleId')
        return render(request, 'accepted.html', {'userName': userName, 'confidence': confidence, 'roleId': roleId})


class RejectedPageView(TemplateView):
    def get(self, request, **kwargs):
        userName = request.GET.get('userName')
        confidence = request.GET.get('confidence')
        roleId = request.GET.get('roleId')
        return render(request, 'rejected.html', {'userName': userName, 'confidence': confidence, 'roleId': roleId})


class GetRolesView(TemplateView):
    def get(self, request, **kwargs):
        roles = []
        for role in Roles.objects.all().only('role', 'url'):
            roles.append({'role': role.role, 'url': static(role.url)})
        return JsonResponse({'status': 'true', 'roles': roles})


class RegisterView(TemplateView):
    __reg = Register()

    def update_database(self, user_name, role_instance, bcs):
        # Train classifier using biocapsules
        classifier = self.__reg.register_classifier(bcs)

        # Convert biocapsules and classifier to binary to save in database
        classifier_binary = pickle.dumps(classifier)
        bcs_binary = pickle.dumps(bcs)

        # Save new user into database
        new_user = Users(user_name=user_name, role=role_instance, bio_capsule=bcs_binary, classifier=classifier_binary)
        new_user.save()
        print('register - database updated')
        return

    def post(self, request, **kwargs):
        ### Get POST data form
        try:
            # Extract data
            user_name = request.POST.get('userName', '')
            role = request.POST.get('role', '')

            # Check username exists
            user_found = Users.objects.filter(username=user_name)
            if user_found.count() < 1:
                return JsonResponse({'status': 'not_taken', 'error': 'Username ' + user_name + ' has not been registered.'})

            # Check if already registered
            user_instance = user_found[0]
            dynaswap_user = DynaSwapUsers.objects.filter(dynaswap_user_id=user_instance.user_id, role=role)
            if dynaswap_user.count() > 0:
                return JsonResponse({'status': 'taken', 'error': user_name + ' already registered as ' + role + ' role.'})

            # Check user_role pair exists
            user_role = UsersRoles.objects.filter(user_id=user_instance.user_id, role=role)
            if user_role.count() < 1:
                return JsonResponse({'status': 'invalid_pair', 'error': user_name + ' cannot be registered as ' + role + ' role.'})

            images = []
            for key, value in request.POST.items():
                if key[:5] == 'image':
                    format, imgstr = value.split(';base64,')
                    ext = format.split('/')[-1]
                    image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
                    images.append(image)
        except Exception as e:
            return JsonResponse({'status': 'false', 'error': 'POST data error. ' + str(e)})

        ### Process POST data form
        try:
            # Convert submited images to biocapsules
            bcs = np.empty((0, 514))
            for image in images:
                stream = image.file
                image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
                bc = self.__reg.register_image(image, role)
                bcs = np.vstack([bcs, bc])
        except Exception as e:
            return JsonResponse({'status': 'image', 'error': 'Invalid input image. ' + str(e)})

        ### Update database
        try:
            t = threading.Thread(target=self.update_database, args=(user_name, role_instance, bcs))
            t.setDaemon(True)
            t.start()
        except Exception as e:
            return JsonResponse({'status': 'error'})

        return JsonResponse({'status': 'true'})


class AuthenticateView(TemplateView):
    __reg = Register()
    __auth = Authenticate()

    def update_database(self, user_found, bc, bcs):
        # Update user bcs
        bc = np.reshape(np.append(bc, 1.), (1, 513))
        bcs = np.vstack([bcs, bc])
        bcs_binary = pickle.dumps(bcs)
        user_found.bio_capsule = bcs_binary

        # Update user classifier
        classifier = self.__reg.register_classifier(bcs)
        classifier_binary = pickle.dumps(classifier)
        user_found.classifier = classifier_binary

        # Save updated user into database
        user_found.last_authenticated = timezone.now()
        user_found.save()
        print('authenticate - database updated')
        return

    def post(self, request, **kwargs):
        ### Get POST data form
        try:
            # Extract data
            user_name = request.POST.get('userName', '')
            role = request.POST.get('role', '')
            temp_image = request.POST.get('image', '')
            format, imgstr = temp_image.split(';base64,')
            ext = format.split('/')[-1]
            image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
        except Exception as e:
            return JsonResponse({'status': 'false', 'error': 'POST data error. ' + str(e)})

        ### Check if username is already registered
        user_found = Users.objects.filter(user_name=user_name)
        if user_found.count() < 1:
            return JsonResponse({'status': 'not_taken', 'error': 'Username ' + user_name + ' has not been registered.'})
        user_found = user_found[0]

        ### Process POST data form
        try:
            # Get role info corresponding to chosen role
            role_instance = Roles.objects.filter(role=role)[0]
            role_id = role_instance.id

            # Convert submited images to biocapsules
            stream = image.file
            image = cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
            bc = self.__auth.authenticate_image(image, role_id)
        except Exception as e:
            return JsonResponse({'status': 'image', 'error': 'Invalid input image. ' + str(e)})

        ### Perform classification, Update database
        try:
            # Get bcs and classifier corresponding to username
            bcs = pickle.loads(user_found.bio_capsule)
            classifier = pickle.loads(user_found.classifier)

            # Perform authentication
            classification, prob = self.__auth.authenticate_classifier(bc, classifier)
            prob_string = str(np.around((prob * 100.), 3)) + '%'

            # Classification outcome
            if not classification:
                return JsonResponse({'status': 'false', 'userName': user_name, 'confidence': prob_string})

            # Update user with authentication time, new bc, new classifier
            if prob > 0.70:
                t = threading.Thread(target=self.update_database, args=(user_found, bc, bcs))
                t.setDaemon(True)
                t.start()

            # Update user with authentication time
            else:
                user_found.last_authenticated = timezone.now()
                user_found.save()
        except Exception as e:
            return JsonResponse({'status': 'error'})

        return JsonResponse({'status': 'true', 'userName': user_name, 'confidence': prob_string})


class GetUserView(TemplateView):
    def get(self, request, **kwargs):
        try:
            user_name = request.GET.get('userName')
            user_found = Users.objects.filter(username=user_name)
            if user_found.count() < 1:
                return JsonResponse({'status': 'unknown'})

            return JsonResponse({'status': 'found'})
        except Exception as e:
            return JsonResponse({'status': 'exception', 'error': str(e)})
