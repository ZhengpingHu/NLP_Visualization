# NLP_Visualization
Course Project for CSCI6612 Data Visualization

## Author
Ye Wang

Zhengping Hu

## Project Proposal
The purpose of this project is to use NLP technology to analyze NYT article comments within a specific time period, determine the user's emotions and the associated words they are interested in, and provide media practitioners with more in-depth analysis and clear visualization content.


At the same time, we use the GNN model to analyze the relevance of various words in user comments, and finally combine the results with the local LLM to provide authors with keyword-based writing outlines, which can greatly improve the author's writing efficiency and increase the author's article effect in a targeted manner.


## Dataset
Data source: [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) (from Kaggle)

## OS requirement
[Ubuntu 22.04](https://releases.ubuntu.com/jammy/)

## Directory Setup
Create the following folders under the `Data_Analysis_Demo` folder:

- **`dataset`**: Place all decompressed datasets here.
- **`models`**: Trained models will be saved in this folder.

## Starting from scratch
Please use the file step by step under the `Data_Analysis_Demo` folder when you start at new dataset.

Or you can directly start the demo, models already saved in `Data_Analysis_Demo\models` folder.

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
3. Install the Redis server (for Ubuntu)
   ```bash
   sudo apt-get update
   sudo apt-get install redis-server
4. Run the Redis server
   ```bash
   sudo service redis-server start
5. Make sure you install the Celery server
   ```bash
   pip install "celery[redis]"
6. Verify Redis and Celery setup:
    ```bash
    celery -A backend worker --loglevel=info
7. Modify the [absolute address](https://github.com/ZhengpingHu/NLP_Visualization/blob/main/Website_Demo/backend.py#L59-L68) before you start the project.


## Project Structure
NLP_Visualization

    .DS_Store  
    .gitignore  
    README.md  
    requirements.txt
    
    Data_Analysis_Demo/  
        dataset/  
        demo/  
            step1_data_pre-cleaning.py  
            step2_data_selection.py  
            step3_GNN_training.py  
            step4_GNN_replay.py
        models/
            gnn_model.pth
            graph_data.pth
            index_to_word.pkl
            word_to_index.pkl

    Project_Proposal/  
        Project.pdf   

    Website_Demo/  
        backend.py  
        celery_worker.py
        GNN.py
        static/   
            css/  
            fonts/  
            images/  
        templates/  
            index.html 
            explor.html  
            visual.html
            conclusion.html
            submit.html
            processing.html
            result.html   

    Analytics/  
        NLP_Visualization_Comments_step1.ipynb  
        NLP_Visualization_Comments_step1.ipynb - Colab.pdf  
        NLP_Visualization_Comments_step2.ipynb  
        NLP_Visualization_Comments_step2.ipynb - Colab.pdf


## Usage
1. Run `start.py` to start the server.
   ```bash
   python3 start.py

2. Default accessing location for the Demo website: [Here](http://192.168.0.246:6612) in `Website_Demo/backend.py`. (Modify [Here](https://github.com/ZhengpingHu/NLP_Visualization/blob/main/Website_Demo/backend.py#L375))

## Hint
Due to capacity limitations, all images in the final Demo were deleted to reduce the size of the compressed package, and the project visualization launched from the compressed package is incomplete.
To solve this problem, you can find all the content in the previous project Github link.

## References
This project relies on the following key libraries and tools:

1. **[Pandas](https://pandas.pydata.org/)**  
   A powerful Python library for data manipulation and analysis. Used extensively for handling and cleaning the dataset, as well as merging and filtering features.

2. **[NumPy](https://numpy.org/)**  
   A fundamental package for numerical computations in Python. Utilized for handling large arrays and performing mathematical operations.

3. **[spaCy](https://spacy.io/)**  
   An advanced Natural Language Processing library. Applied for part-of-speech tagging.

4. **[VADER](https://github.com/cjhutto/vaderSentiment)**  
   A lexicon and rule-based sentiment analysis tool. Used for assigning sentiment scores (positive, negative, neutral) to user comments.

5. **[Matplotlib](https://matplotlib.org/)**  
   A comprehensive library for creating static visualizations. Used to generate sentiment distribution graphs and other visualizations.

6. **[WordCloud](https://github.com/amueller/word_cloud)**  
   A Python library for generating word clouds. Used to visualize keyword distributions across sentiment categories.

7. **[Flask](https://flask.palletsprojects.com/)**  
   A lightweight web framework for building the project’s backend and serving the visualization results.

8. **[Dash](https://plotly.com/dash/)**  
   A framework for building analytical web applications. Utilized to create dashboards for interactive data exploration.

9. **[PyTorch](https://pytorch.org/)**  
   A deep learning framework used to build and train the Graph Neural Network (GNN) for sentiment-driven keyword relationships.

10. **[Torch Geometric](https://pytorch-geometric.readthedocs.io/)**  
    An extension library for PyTorch. Used to implement and train the GNN for graph-based modeling.

11. **[Redis](https://redis.io/)**  
    An in-memory data structure store used as a message broker to manage intermediate data and task queues.

12. **[Celery](https://docs.celeryproject.org/)**  
    A distributed task queue. Used for scheduling and managing long-running computational tasks asynchronously.

13. **[OpenAI](https://chatgpt.com/)**  
    Much help in understanding concepts was by chatGPT, but no direct code was copied from chatGPT.







