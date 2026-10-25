"""
URL configuration for Loyiha project — Telekom Churn Tahlili
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from telecom import views as dv

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dv.index, name='index'),
    path('index/', dv.index, name='index'),
    path('database/', dv.database, name='database'),
    path('algorithm/', dv.algorithm, name='algorithm'),
    path('analysis/', dv.analysis, name='analysis'),
    path('klassterlash/', dv.klassterlash_view, name='klassterlash'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)