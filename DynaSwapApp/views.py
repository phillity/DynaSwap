import base64
import os
import time
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
        return JsonResponse({"status": "true", "roles" : [
                { "role" : "President", "url" : static('/images/rs_1.jpg') }, 
                { "role" : "CEO", "url" : static('/images/rs_2.jpg') },
                { "role" : "Manager", "url" : static('/images/rs_3.jpg') },
                { "role" : "Worker", "url" : static('/images/rs_4.jpg') },
            ]})

class UploadImageView(TemplateView):
    def post(self, request, **kwargs):
        try:
            imageRaw = request.POST.get("image", "")
            format, imgstr = imageRaw.split(';base64,') 
            ext = format.split('/')[-1] 
            image = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)

            userName = request.POST.get("userName", "")
            role = request.POST.get("role", "")

            # Save the file
            fs = FileSystemStorage()
            image_file_name = fs.save(userName + "_" + time.strftime("%Y%m%d%H%M%S") + ".jpg", image)
            uploaded_file_url = fs.url(image_file_name)
            return JsonResponse({"status":"true", "filePath": uploaded_file_url})
        except Exception as e:
            return JsonResponse({"status" : "false", "error" : str(e)})
