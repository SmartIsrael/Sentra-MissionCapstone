"""
WSGI config for smartel project.

It exposes the WSGI callable as a module-level variable named ``application``.
For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""
import os
import sys
from django.core.wsgi import get_wsgi_application

# Adjust the sys.path so that the correct location is included
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'smartel')))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartel.settings')

application = get_wsgi_application()
