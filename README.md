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

Project Structure:

NLP_Visualization
|   .DS_Store
|   .gitignore
|   README.md
|   requirements.txt
|
+---Data_Analysis_Demo
|   +---dataset
|   +---demo
|   |       data_cleaning.py
|   |       model_output.py
|   |       model_training.py
|   |
|   +---models
|   +---old_testing_files
|   |       dataset_cleaning.py
|   |       GNN_test.py
|   |       Inference_model.py
|   |       mini_set_cleaning.py
|   |       naive_bayes_training.py
|   |       pre_cleaning.py
|   |       single_article_visualization.py
|   |       step1_data_cleaning.py
|   |
|   \---temp
+---Project_Proposal
|   |   Project.pdf
|   |
|   \---Recordings
|           record 1.m4a
|
\---Website_Demo
    |   backend.py
    |   LLM_query_test.py
    |
    +---static
    |   |   about.html
    |   |   contact.html
    |   |   myplot.html
    |   |   myplot.png
    |   |   portfolio.html
    |   |   team.html
    |   |
    |   +---css
    |   |
    |   +---fonts
    |   |
    |   \---images
    |
    \---templates
            index.html
            result.html
            specific.html
            submit.html
            team_intro.html
            wordcloud.html
