#!/bin/sh

# Check the SERVER_TYPE environment variable to decide whether to run Gunicorn or Daphne
if [ "$SERVER_TYPE" = "daphne" ]; then
    echo "Starting Daphne server..."
    exec daphne -b 0.0.0.0 -p 8001 backend.asgi:application
else
    echo "Starting Gunicorn server..."
    exec gunicorn --bind 0.0.0.0:8000 backend.wsgi:application --timeout 15 --preload
fi
