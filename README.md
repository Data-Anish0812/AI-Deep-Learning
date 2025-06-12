# Customer Churn Prediction
In the churn prediction analysis, both an optimized neural network (NN) and a simple NN were implemented to evaluate their effectiveness in predicting customer churn. The optimized NN included enhancements such as hyperparameter tuning, dropout regularization, and early stopping, while the simple NN was built with a basic architecture and default settings. After training both models on the same dataset, the optimized NN demonstrated higher accuracy compared to the simple NN, showing that optimization techniques significantly improve model performance. This comparison highlights the value of fine-tuning in achieving more accurate churn predictions.
# Movie Recommendation System
In **Book Recommendation System**, Performed data preprocessing on movie metadata by cleaning the dataset, handling missing values, and selecting key columns such as genres, overview, keywords, cast, and crew to support recommendation logic. Built a content-based recommendation system that calculates movie similarity using cosine similarity on features extracted with CountVectorizer. This method recommends movies based on similarities in genres, descriptions, and other textual attributes, effectively aligning suggestions with user preferences.
# Musical Instrument Detection using CNN
**Musical Instrument Detection using CNN** focuses on classifying musical instruments using a Convolutional Neural Network (CNN) model trained on a labeled image dataset. The dataset was loaded and preprocessed using TensorFlow's image pipeline, including data augmentation and optimization with caching, prefetching, and shuffling. A custom CNN model was built with increasing convolutional layers (32→64→128), followed by dense layers for classification. To enhance training efficiency and performance, callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau were used. The model achieved strong performance in just a few epochs, demonstrating an end-to-end deep learning workflow from data loading to model evaluation.
# Time series analysis on Answered calls
**Time series analysis on Answered calls** on Answered calls explores time series forecasting using statistical and machine learning techniques. The notebook includes a comprehensive workflow starting from data loading and visualization to stationarity checks, decomposition, and model implementation. It covers Autoregressive (AR), Moving Average (MA), ARIMA, and SARIMA models, along with performance evaluation using metrics like RMSE. The project demonstrates practical steps to analyze and forecast temporal data, making it a valuable resource for anyone interested in predictive analytics and time series modeling.
# Sppech to Text Conversion
**Sppech to Text Conversion** demonstrates speech-to-text conversion using OpenAI's Whisper library. A simple audio file is processed and transcribed into text with high accuracy, showcasing Whisper's powerful multilingual and robust transcription capabilities. The implementation is straightforward, making it easy to adapt for various audio processing tasks such as voice assistants, transcription services, or language learning tools.
# Dmart reviews analysis using NLP
This project focuses on **NLP cleaning and sentiment analysis** of D-Mart reviews. Using the NLTK library, stopwords were removed, and popular words were extracted to refine the text data. Sentiment analysis was performed to classify reviews into **positive, negative, and neutral categories**, which were then concatenated for a comprehensive overview. This analysis helps in understanding customer opinions and extracting valuable insights from the reviews.
# Hotel review sentiment analysis
This project focuses on **NLP preprocessing and sentiment analysis** of hotel reviews. Using the NLTK library, stopwords were removed, and frequently occurring words were identified to enhance text refinement. **Exploratory Data Analysis (EDA)** was carried out to uncover the most common positive and negative words, offering a deeper understanding of guest sentiment. To further enhance visualization, the **WordCloud** library was utilized to display the most repeated words across the reviews. Sentiment analysis was then conducted to classify reviews into **positive, negative, and neutral categories**, which were combined for a holistic evaluation. This analysis aids in understanding guest feedback and deriving valuable insights from the reviews.
# Book Recommendation system
**Book Recommendation System** built using K-Nearest Neighbors (KNN) algorithm. It involves data preprocessing techniques such as Standard Scaling and Min-Max Scaling, along with feature engineering to improve the recommendation quality. A custom function has been created which, when given a book title as input, returns the top 5 most similar book recommendations based on reader preferences and book features. The system aims to provide quick and relevant suggestions to enhance the user’s reading experience.
# Sentiment Analysis using LSTM
This project focuses on Sentiment Analysis using an LSTM (Long Short-Term Memory) neural network. The model is trained on a labeled dataset of text comments to classify sentiments into three categories. The pipeline includes data cleaning, tokenization, padding of sequences to a uniform length, and splitting the data for training and testing. A Keras LSTM model with an embedding layer is used to capture the context and sequential dependencies in the text. The model is compiled with the Adam optimizer and trained using sparse categorical crossentropy loss, with early stopping to prevent overfitting. This project demonstrates how deep learning can effectively understand and classify human sentiments from natural language.





















