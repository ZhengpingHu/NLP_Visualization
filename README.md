# NLP_Visualization
Course Project for CSCI6612 Data Visualization

## Dataset
Data source: [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) (from Kaggle)

## Directory Setup
Create the following folders under the `Data_Analysis_Demo` folder:

- **dataset**: Place all decompressed datasets here.
- **models**: Trained models will be saved in this folder.
- **temp**: Temporary files generated during training will be stored here.

## Backend Requirements
`backend.py` requires the llama 3.2 model. Follow the installation instructions provided [here](https://ollama.com/library/llama3.2).

## Templates
You can download free templates for the web interface from [HTML Design](https://html.design/).

## Environment Setup
1. Install all pip requirements:
   ```bash
   pip install -r requirements.txt
2. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm

## Project Structure
NLP_Visualization

    .DS_Store  
    .gitignore  
    README.md  
    requirements.txt
    
    Data_Analysis_Demo/  
        dataset/  
        demo/  
            data_cleaning.py  
            model_output.py  
            model_training.py  
        models/  
        old_testing_files/  
            dataset_cleaning.py  
            GNN_test.py  
            Inference_model.py  
            mini_set_cleaning.py  
            naive_bayes_training.py  
            pre_cleaning.py  
            single_article_visualization.py  
            step1_data_cleaning.py  
        temp/  

    Project_Proposal/  
        Project.pdf  
        Recordings/  
            record 1.m4a  

    Website_Demo/  
        backend.py  
        LLM_query_test.py  
        static/  
            about.html  
            contact.html  
            myplot.html  
            myplot.png  
            portfolio.html  
            team.html  
            css/  
            fonts/  
            images/  
        templates/  
            index.html  
            result.html  
            specific.html  
            submit.html  
            team_intro.html  
            wordcloud.html

## Usage
Add your datasets in Data_Analysis_Demo/dataset/.

Run backend.py to start the server.

Open index.html in a browser to access the visualization dashboard.

Default accessing location for the Demo website:[Here](https://192.168.0.246:6612).
