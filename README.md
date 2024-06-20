# OtakuViz: Explore Anime Insights

OtakuViz is a powerful and interactive web application designed to provide comprehensive insights into the anime world. Built with Streamlit, OtakuViz offers data visualization, anime recommendations, and score predictions to help users discover and explore anime content more effectively. This README file outlines the features of OtakuViz, its usefulness to different users, tools used, algorithms description, setup instructions, and more.
Table of Contents
1. Features
2. Usefulness to Users and Stakeholders
3. Example Scenarios
4. Tools Used
5. Algorithms Description
6. Setup and Installation
7. Usage
8. Evaluation Metrics
9. Contributing
10. License
Features
1. Data Visualization:
    * Visualize anime data by genre, type, status, source, and rating.
    * Different plot types: bar plots, line plots, and pie charts.
2. Additional Analysis:
    * Insights and analysis on various aspects of anime data.
3. Recommendation System:
    * Get anime recommendations based on selected genres and types.
4. Text-Based Recommendation:
    * Find anime recommendations based on synopsis similarity using cosine similarity.
5. Score Prediction:
    * Predict the score of an anime based on attributes like favorites, members, award-winning status, episodes, and duration.
Usefulness to Users and Stakeholders
Users
Anime Fans:
* Discover New Anime: Users can get personalized recommendations based on their favorite genres and types.
    * Example: A user who loves action and adventure anime can find new titles matching these genres.
Students and Researchers:
* Data Analysis: Provides tools to visualize and analyze trends in anime data for research purposes.
    * Example: A student studying the popularity trends of different anime genres over time can use the data visualization features.
Content Creators:
* Content Planning: Helps content creators understand what type of anime is currently popular or award-winning.
    * Example: A YouTuber creating anime review videos can identify trending anime to review next.
Stakeholders
Anime Production Companies:
* Market Research: Analyze viewer preferences and trends to make informed decisions about future productions.
    * Example: A production company can explore which genres have been most popular to guide their next project.
Anime Streaming Platforms:
* User Engagement: Provide better recommendations to keep users engaged with the platform.
    * Example: A streaming service can integrate the recommendation system to suggest anime to users based on their watch history.
Advertisers and Marketers:
* Targeted Campaigns: Design marketing campaigns targeting specific anime genres or types.
    * Example: An advertiser can plan a campaign for action anime viewers, knowing the popularity and engagement levels of that genre.
Example Scenarios
1. Anime Fan:
    * Scenario: An anime fan named Alex wants to find new anime similar to "Naruto".
    * Action: Alex uses the Text-Based Recommendation feature, selects "Naruto", and receives a list of similar anime recommendations.
2. Student/Researcher:
    * Scenario: A student named Jamie is conducting research on the evolution of anime genres.
    * Action: Jamie uses the Data Visualization feature to create line plots showing the popularity of different genres over the years.
3. Content Creator:
    * Scenario: A YouTuber named Sam needs ideas for the next anime review video.
    * Action: Sam uses the Recommendation System to find trending anime in the "Fantasy" genre and chooses one to review.
4. Production Company:
    * Scenario: A production company executive named Kim wants to know which types of anime have high scores.
    * Action: Kim uses the Score Prediction feature to analyze attributes of high-scoring anime and guides the production team accordingly.
Tools Used
* Streamlit: Framework for building and deploying the web application interface.
* Python Libraries:
    * Pandas: Data manipulation and analysis.
    * NumPy: Mathematical operations on arrays.
    * NLTK (Natural Language Toolkit): Text preprocessing for synopsis-based recommendations.
    * Scikit-learn: Machine learning tools for data preprocessing, modeling, and evaluation.
    * Matplotlib and Seaborn: Data visualization tools for creating plots.
Algorithms Description
Cosine Similarity
* Purpose: Used in the Text-Based Recommendation feature to measure the similarity between anime synopses.
* Implementation:
    * Anime synopses are preprocessed to remove punctuation, convert to lowercase, remove stopwords, and perform stemming.
    * Vectorization using CountVectorizer to convert text into numerical vectors.
    * Cosine similarity calculation to determine the similarity between the synopses vectors.
Random Forest Regression
* Purpose: Used in the Score Prediction feature to predict anime scores based on attributes.
* Implementation:
    * Data preprocessing including one-hot encoding categorical variables and handling missing values.
    * Training the Random Forest Regressor on training data to learn patterns and relationships.
    * Predicting anime scores using the trained model.
Setup and Installation
Prerequisites
* Python 3.7 or higher
* pip (Python package installer)
Installation Steps
1. Clone the Repository: bashCopy code  git clone https://github.com/yourusername/Otakuviz.git
2. cd Otakuviz
3.   
4. Install Required Packages: bashCopy code  pip install -r requirements.txt
5.   
6. Download NLTK Data: pythonCopy code  import nltk
7. nltk.download('stopwords')
8. nltk.download('punkt')
9.   
10. Run the Application: bashCopy code  streamlit run app.py
11.   
12. Access the Application: Open your web browser and go to http://localhost:8501.
Usage
Data Visualization
1. Select the parameter you want to visualize (Genre, Type, Status, Source, Rating).
2. Choose a specific value for the selected parameter.
3. Select the plot type (Bar Plot, Line Plot, Pie Chart).
4. The application displays the plot based on your selections.
Recommendation System
1. Select genres and types for which you want recommendations.
2. The application provides a list of anime that match the selected criteria.
Text-Based Recommendation
1. Select an anime from the dropdown menu.
2. The application provides a list of similar anime based on synopsis similarity.
Score Prediction
1. Enter values for Favorites, Members, Award Winning, Episodes, and Duration.
2. The application predicts and displays the expected score for the anime.
Evaluation Metrics
The performance of the score prediction model is evaluated using:
* Root Mean Squared Error (RMSE): Measures the square root of the average squared differences between predicted and actual scores.
* Mean Absolute Error (MAE): Measures the average absolute differences between predicted and actual scores.
Privacy and Data Handling: OtakuViz uses publicly available anime data for analysis and recommendations. However, privacy concerns may arise if users' personal information or preferences are linked with their usage data. We ensure that only anonymized data is used for analysis to protect user privacy.
Bias and Fairness: The recommendation system in OtakuViz relies on algorithms that may exhibit bias based on the data they are trained on. Efforts have been made to mitigate bias, but users should be aware that algorithmic recommendations may not always be fair or inclusive.
Algorithmic Accountability: While algorithms power many features in OtakuViz, they are not infallible. There is ongoing work to ensure that algorithms used in the app are accountable, transparent, and continuously improved to minimize unintended consequences.
