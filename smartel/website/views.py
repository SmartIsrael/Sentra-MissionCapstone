from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "website/index.html")


def base(request):
    return render(request, "website/base.html")


def career(request):
    return render(request, "website/portfolio-details.html")


def portfolio_details(request):
    return render(request, "website/portfolio-details.html")


def contact(request):
    return render(request, "website/contact-us.html")


def product(request):
    return render(request, "website/product.html")


def about(request):
    return render(request, "website/about-us.html")
