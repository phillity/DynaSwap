from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from DynaSwapApp import views

urlpatterns = [
    url(r'^$', views.HomePageView.as_view(), name='home_page'),
    url(r'^register_page/$', views.RegisterPageView.as_view(), name='register_page'),
    url(r'^authenticate_page/$', views.AuthenticatePageView.as_view(), name='authenticate_page'),
    url(r'^get_roles/$', views.GetRolesView.as_view(), name='get_roles'),
    url(r'^registration/$', views.RegisterView.as_view(), name='registration'),
    url(r'^authentication/$', views.AuthenticateView.as_view(), name='authentication'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)