#!/bin/sh
export FLASK_APP=./app/flask_app.py
#export FLASK_DEBUG=1
source venv/bin/activate
python3 -m flask run 