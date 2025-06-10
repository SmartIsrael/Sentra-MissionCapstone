#!/bin/bash
source venv/bin/activate
exec gunicorn smartel.wsgi:application --bind 127.0.0.1:8090
