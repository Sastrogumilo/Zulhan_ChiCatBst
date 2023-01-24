from django.urls import path, include
from . import views

app_name = 'administrator'

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset/', views.dataset, name='dataset'),
    path('tentang/', views.tentang, name='tentang'),
    path('EDA/', views.EDA, name='EDA'),
    path('encode/', views.encode, name='encode'),
    path('split_data_chi2/', views.SplitData, name='split_data_chi2'),
    #path('klasifikasi/', views.klasifikasi, name='klasifikasi'),
    path('Catboost/', views.Catboost, name='Catboost'),
    path('Catboost_Chi2/', views.Catboost_Chi2, name='Catboost_Chi2'),
    path('hasil_catboost_chi2/', views.hasil_catboost_chi2, name='hasil_catboost_chi2'),
    #path('hasilsvmrbf/', views.hasilsvmrbf, name='hasilsvmrbf'),
    #path('SVMRBFIG/', views.SVMRBFIG, name='SVMRBFIG'),
    path('live_test/', views.live_test, name="live_test"),
    path('live_test_hasil/', views.live_test_hasil, name="live_test_hasil"),
    path('live_test2/', views.live_test2, name="live_test2"),
    path('live_test_hasil2/', views.live_test_hasil2, name="live_test_hasil2")
]
