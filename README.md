# Customer Churn Prediction
In the churn prediction analysis, both an optimized neural network (NN) and a simple NN were implemented to evaluate their effectiveness in predicting customer churn. The optimized NN included enhancements such as hyperparameter tuning, dropout regularization, and early stopping, while the simple NN was built with a basic architecture and default settings. After training both models on the same dataset, the optimized NN demonstrated higher accuracy compared to the simple NN, showing that optimization techniques significantly improve model performance. This comparison highlights the value of fine-tuning in achieving more accurate churn predictions.
# Movie Recommendation System
In **Book Recommendation System**, Performed data preprocessing on movie metadata by cleaning the dataset, handling missing values, and selecting key columns such as genres, overview, keywords, cast, and crew to support recommendation logic. Built a content-based recommendation system that calculates movie similarity using cosine similarity on features extracted with CountVectorizer. This method recommends movies based on similarities in genres, descriptions, and other textual attributes, effectively aligning suggestions with user preferences.
# Musical Instrument Detection using various algorithms
**Musical Instrument Detection using various algorithms** and transfer learning focuses on classifying images of musical instruments through deep learning models trained on a labeled dataset. The data was loaded and preprocessed using TensorFlow’s image pipeline, incorporating augmentation, caching, prefetching, and shuffling for performance optimization. Initially, a custom CNN was built with progressively deeper convolutional layers (32→64→128) followed by dense layers. To further enhance accuracy and efficiency, pre-trained models like **MobileNetV2** and **EfficientNetB7** were integrated using transfer learning. These models were fine-tuned on the dataset to leverage rich feature extraction. The training process utilized callbacks such as EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for stability and control. Each model was thoroughly evaluated using metrics like **F1 score**, **precision**, and **recall**, ensuring a robust comparison and highlighting the effectiveness of the models in real-world multi-class classification.
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






















