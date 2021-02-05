## Set up
* git submodule update --init
* Download model https://drive.google.com/open?id=1_2CCb_qsA1egT5c2s0ABuW3rQCDOLvPq and place in flask/models/
* pip3 install virtualenv
* cd flask
* virtualenv venv
* source venv/bin/activate
* pip3 install -r requirements.txt

## Starting Flask server
* cd flask
* export FLASK_APP=main.py
* flask run
* Navigate to http://localhost:5000/

## Debugging Flask server (assuming)
* Run `python -m debugpy --listen 0.0.0.0:5678 -m flask run --host=0.0.0.0 --reload`
* Go to the `Run` sidebar in VSCode and run `Python: Remote`
* Set breakboint in main.py
