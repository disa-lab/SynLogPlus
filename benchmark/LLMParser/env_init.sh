#!/bin/bash

pip install -r requirements.txt
git clone https://github.com/thunlp/OpenPrompt
pushd openprompt
patch -p1 < ../openprompt.patch
pip install -r requirements.txt
pip install pandas
pip install scikit-learn
python setup.py install
popd
