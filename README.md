## About ##
Project to ocr vietnamese text in image use STAR net for text recognition
## Require ##
Docker (if you want to use dockerfile), python3.8, pip
## Usage ##
```bash
# first you need to install dependencies for this project, you can use docker or install directly
# use docker
docker build . -t ocr_service:v0
docker run --name <docker name> -v <path to root folder of project>:/project/ -it -p 8001:8001 ocr_service:v0 /bin/bash

# install directly
# you can use virtual enviroment (not requires you can install depecdencies directly if dont want use virtual env)
pip3 install virtualenv
python3 -m venv myEnv
source myEnv/bin/activate
# then install dependencies
pip install -r requirements.txt

# run
python3 predict.py
# run on localhost:8001 
```