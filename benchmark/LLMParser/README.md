# LLMParser: An Exploratory Study on Using Large Language Models for Log Parsing

See original at https://github.com/zeyang919/LLMParser

### Steps


```
# First, create and activate python virtual environment

# Second, install dependencies
pip install -r requirements.txt
git clone https://github.com/thunlp/OpenPrompt
pushd openprompt
patch -p1 < ../openprompt.patch
pip install -r requirements.txt
pip install pandas
pip install scikit-learn
python setup.py install
popd

# Lastly, download the LLM
pushd LLMs/
sh LLMs/flan-t5-base.sh
```

### Run

```
python data_sampling --full --shot 50
python LLMParser.py --full --train_percentage 50_2000h --num_epochs 50 --model 'flan-t5-base'
```
