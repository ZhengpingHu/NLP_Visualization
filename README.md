# NLP_Visualization
Course Project for CSCI6612 Data Visualization

Dataset (From Kaggle):

https://www.kaggle.com/datasets/aashita/nyt-comments


Tips:

Create three new folders under the Data_Analysis_Demo folder:

**dataset** (Put all the decompressed databases in this folder)

**models** (The trained models will be stored here)

**temp** (The temporary files during training will be stored here)


backend.py requires llama 3.2 model, installation process at: 

https://ollama.com/library/llama3.2

for Templates:

free download from website: 

https://html.design/

To setup the environment:

1. for all pip requirements

copy and paste the command into terminal:

**pip install -r requirements.txt**

2. for spaCy (the local English model):
   
copy and paste the command into terminal:

**python -m spacy download en_core_web_sm** 
