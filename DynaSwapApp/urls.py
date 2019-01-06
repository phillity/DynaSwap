from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from DynaSwapApp import views
from django.views.generic import TemplateView

urlpatterns = [
    url(r'^$', views.HomePageView.as_view(), name='home'),
    url(r'^register/$', views.RegistrationView.as_view(), name='register'),
    url(r'^authenticate/$', views.AuthenticationView.as_view(), name='authenticate'),
    url(r'^getroles/$', views.GetRolesView.as_view(), name='get_roles'),
    url(r'^upload/image/$', views.UploadImageView.as_view(), name='upload_image'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)