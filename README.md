This project involves creating a web application using Streamlit for performing sentiment analysis on YouTube comments. 
The main functionalities of the project include fetching YouTube comments, preprocessing the text, and analyzing the sentiments of the comments. 
Here's a detailed description of each part of the project:

Libraries and Tools Used
Streamlit: A framework to build and share data applications.
Requests: For making HTTP requests to the YouTube API.
Pandas: For data manipulation and analysis.
NLTK: The Natural Language Toolkit, used for text preprocessing.
Transformers (Hugging Face): A library for natural language processing tasks, used here for sentiment analysis.
Seaborn: For data visualization

Key Features : 
Fetch YouTube Comments: Fetches comments from a given YouTube video using the YouTube Data API.
Text Preprocessing: Cleans and preprocesses the comments to prepare them for sentiment analysis.
Sentiment Analysis: Uses a pre-trained DistilBERT model to analyze the sentiment of each comment.
Visualization: Displays the results in a table and visualizes the sentiment distribution using a bar chart.
