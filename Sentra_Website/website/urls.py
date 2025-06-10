from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.index, name="index"),
    path("base/", views.base, name="base"),
    path("career/", views.career, name="career"),
    path("portfolio-details/", views.portfolio_details, name="portfolio-details"),
    # path('services/', views.services, name='services'),
    path("contact/", views.contact, name="contact"),
    path("product/", views.product, name="product"),
    path("about/", views.about, name="about"),
]
