## Set up
* git submodule update --init
* Download model https://drive.google.com/open?id=1_2CCb_qsA1egT5c2s0ABuW3rQCDOLvPq and place in flask/models/
* pip3 install virtualenv
* cd flask
* virtualenv venv

## Starting Flask server
* cd flask
* source venv/bin/activate
* pip3 install -r requirements.txt
* export FLASK_APP=main.py
* flask run
* Navigate to http://localhost:5000/
