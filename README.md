# OtakuViz: Explore Anime Insights

OtakuViz is an interactive web application designed to provide comprehensive insights into the anime world. Built with Streamlit, OtakuViz offers data visualization, anime recommendations, and score predictions to help users discover and explore anime content more effectively. This README file outlines the features of OtakuViz, its usefulness to different users, tools used, algorithms description, and more.

The data used in the OtakuViz Web App is extracted from a dataset in Kaggle which contains vast information about anime.
Link for dataset: https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data?select=anime-dataset-2023.csv

Link for the OtakuViz app: https://otakuviz-x6ag4dpxsxapassa4ymw8h.streamlit.app/

<h2>Table of Contents</h2> 
1. Features <br />
2. Usefulness to Users and Stakeholders <br />
3. Example Scenarios <br />
4. Tools Used <br />
5. Algorithms Description <br />
6. Setup and Installation <br />
7. Usage <br />
8. Evaluation Metrics <br />
9. Contributing <br />
10. License <br />

<h3>Features</h3> 
<b>1. Data Visualization:</b> <br />
    * Visualize anime data by genre, type, status, source, and rating. <br />
    * Different plot types: bar plots, line plots, and pie charts. <br />
<b>2. Additional Analysis:</b> <br />
    * Insights and analysis on various aspects of anime data. <br />
<b>3. Recommendation System:</b> <br />
    * Get anime recommendations based on selected genres and types. <br />
<b>4. Text-Based Recommendation:</b> <br />
    * Find anime recommendations based on synopsis similarity using cosine similarity. <br />
<b>5. Score Prediction:</b> <br />
    * Predict the score of an anime based on attributes like favorites, members, award-winning status, episodes, and duration. <br />
    
<h3>Usefulness to Users and Stakeholders</h3>

<h3>Users</h3>
<b>Anime Fans:</b> <br />
* Discover New Anime: Users can get personalized recommendations based on their favorite genres and types.<br />
    * Example: A user who loves action and adventure anime can find new titles matching these genres.<br />
Students and Researchers:<br />
* Data Analysis: Provides tools to visualize and analyze trends in anime data for research purposes.<br />
    * Example: A student studying the popularity trends of different anime genres over time can use the data visualization features.<br />
Content Creators:<br />
* Content Planning: Helps content creators understand what type of anime is currently popular or award-winning.<br />
    * Example: A YouTuber creating anime review videos can identify trending anime to review next.<br />
    
<h3>Stakeholders</h3>
<b>Anime Production Companies:</b> <br />
* Market Research: Analyze viewer preferences and trends to make informed decisions about future productions. <br />
    * Example: A production company can explore which genres have been most popular to guide their next project. <br />
<b>Anime Streaming Platforms:</b><br />
* User Engagement: Provide better recommendations to keep users engaged with the platform.<br />
    * Example: A streaming service can integrate the recommendation system to suggest anime to users based on their watch history.<br />
<b>Advertisers and Marketers:</b><br />
* Targeted Campaigns: Design marketing campaigns targeting specific anime genres or types.<br />
    * Example: An advertiser can plan a campaign for action anime viewers, knowing the popularity and engagement levels of that genre.<br />
    
<h3>Example Scenarios</h3>
<b>1. Anime Fan:</b> <br />
    * Scenario: An anime fan named Alex wants to find new anime similar to "Naruto". <br />
    * Action: Alex uses the Text-Based Recommendation feature, selects "Naruto", and receives a list of similar anime recommendations. <br />
<b>2. Student/Researcher:</b> <br />
    * Scenario: A student named Jamie is conducting research on the evolution of anime genres. <br />
    * Action: Jamie uses the Data Visualization feature to create line plots showing the popularity of different genres over the years. <br />
<b>3. Content Creator:</b> <br />
    * Scenario: A YouTuber named Sam needs ideas for the next anime review video. <br />
    * Action: Sam uses the Recommendation System to find trending anime in the "Fantasy" genre and chooses one to review. <br />
<b>4. Production Company:</b> <br />
    * Scenario: A production company executive named Kim wants to know which types of anime have high scores. <br />
    * Action: Kim uses the Score Prediction feature to analyze attributes of high-scoring anime and guides the production team accordingly. <br />
    
<h3>Tools Used</h3>
* Streamlit: Framework for building and deploying the web application interface.<br />
* Python Libraries:<br />
    * Pandas: Data manipulation and analysis.<br />
    * NumPy: Mathematical operations on arrays.<br />
    * NLTK (Natural Language Toolkit): Text preprocessing for synopsis-based recommendations.<br />
    * Scikit-learn: Machine learning tools for data preprocessing, modeling, and evaluation.<br />
    * Matplotlib and Seaborn: Data visualization tools for creating plots.<br />
    
<h3>Algorithms Description</h3>
<b>Cosine Similarity</b> <br />
* Purpose: Used in the Text-Based Recommendation feature to measure the similarity between anime synopses.<br />
* Implementation: <br />
    * Anime synopses are preprocessed to remove punctuation, convert to lowercase, remove stopwords, and perform stemming.<br />
    * Vectorization using CountVectorizer to convert text into numerical vectors.<br />
    * Cosine similarity calculation to determine the similarity between the synopses vectors.<br />
    
<b>Random Forest Regression</b><br />
* Purpose: Used in the Score Prediction feature to predict anime scores based on attributes.<br />
* Implementation:<br />
    * Data preprocessing including one-hot encoding categorical variables and handling missing values.<br />
    * Training the Random Forest Regressor on training data to learn patterns and relationships.<br />
    * Predicting anime scores using the trained model.<br />
    
<h3>Usage</h3>
<b>Data Visualization</b> <br />
1. Select the parameter you want to visualize (Genre, Type, Status, Source, Rating). <br />
2. Choose a specific value for the selected parameter. <br />
3. Select the plot type (Bar Plot, Line Plot, Pie Chart). <br />
4. The application displays the plot based on your selections. <br />
<b>Recommendation System</b> <br />
1. Select genres and types for which you want recommendations. <br />
2. The application provides a list of anime that match the selected criteria. <br />
<b>Text-Based Recommendation</b><br />
1. Select an anime from the dropdown menu.<br />
2. The application provides a list of similar anime based on synopsis similarity.<br />
<b>Score Prediction</b> <br />
1. Enter values for Favorites, Members, Award Winning, Episodes, and Duration. <br />
2. The application predicts and displays the expected score for the anime. <br />

<h3>Evaluation Metrics</h3>
<b>The performance of the score prediction model is evaluated using:</b> <br />
<b>* Root Mean Squared Error (RMSE):</b> Measures the square root of the average squared differences between predicted and actual scores. <br />
<b>* Mean Absolute Error (MAE):</b> Measures the average absolute differences between predicted and actual scores. <br />

<h3>Ethical Concerns</h3>
<b>Privacy and Data Handling:</b> OtakuViz uses publicly available anime data for analysis and recommendations. However, privacy concerns may arise if users' personal information or preferences are linked with their usage data. We ensure that only anonymized data is used for analysis to protect user privacy. <br />


<b>Bias and Fairness:</b> The recommendation system in OtakuViz relies on algorithms that may exhibit bias based on the data they are trained on. Efforts have been made to mitigate bias, but users should be aware that algorithmic recommendations may not always be fair or inclusive. <br />

<b>Algorithmic Accountability:</b> While algorithms power many features in OtakuViz, they are not infallible. There is ongoing work to ensure that algorithms used in the app are accountable, transparent, and continuously improved to minimize unintended consequences.
