# Tourism Recommendation and Visit Mode Prediction System

**Overview**

This project is a machine learning–based tourism analytics system that performs two primary tasks:

Visit Mode Prediction (Classification)
Predicts the travel mode of a visitor using historical tourism data.

Attraction Recommendation (Collaborative Filtering)
Recommends attractions based on user similarity using cosine similarity.

The system is implemented in Python using Scikit-learn and deployed with Streamlit.

**Technologies Used**

Python

Pandas

NumPy

Scikit-learn

Joblib

Streamlit

**Project Directory Structure**

The project is organized as follows:

***Root Directory:***

streamlit_app.py → Main Streamlit web application

requirements.txt → List of required Python libraries

README.md → Project documentation

***models/ folder:***

classification.pkl → Trained classification model

user_item.pkl → User-item matrix

similarity.pkl → Cosine similarity matrix

***src/ folder:***

datacleaning.py → Data loading and preprocessing functions

classification_model.py → Script to train classification model

recommendation.py → Script to train recommendation system

**Important:**
All commands must be executed from the root project directory.

**How to Run the Project (Step-by-Step)**
*Step 1: Clone the Repository*

Open terminal and run:

git clone https://github.com/NinceyThomas/tourism-recommendation-system.git

Then navigate into the project folder:

cd tourism-recommendation-system

*Step 2: Create a Virtual Environment*

python -m venv .venv

Activate it:

Windows:
.venv\Scripts\activate

After activation, you should see (.venv) in your terminal.

*Step 3: Install Dependencies*

pip install -r requirements.txt

This installs all required libraries.

*Step 4: Train the Models*

Before running the web application, generate the trained models.

Run the classification model:

python src/classification_model.py

Expected output:

Data loading confirmation

Model training messages

Accuracy score

classification.pkl saved in models folder

Next, run the recommendation model:

python src/recommendation.py

Expected output:

User-item matrix shape

Similarity matrix shape

user_item.pkl and similarity.pkl saved in models folder

Verify that the models/ folder contains the generated .pkl files before proceeding.

*Step 5: Run the Streamlit Application*

Start the application:

streamlit run streamlit_app.py

The application will open automatically in your browser at:

http://localhost:8501

If it does not open automatically, copy the URL displayed in the terminal and paste it into your browser.

**Expected Workflow**

Train models using the scripts inside src/

Confirm model files are generated inside models/

Launch Streamlit app

Use the web interface to:

Predict visit mode

Get attraction recommendations

**Notes for Evaluators**

The project follows modular architecture separating preprocessing, modeling, and UI.

The models are saved using Joblib for reuse within the Streamlit interface.

The recommendation system uses user-based collaborative filtering.

The classification model uses Random Forest for structured feature prediction.

**#Important Note#**
The streamlit app has been deployed and when you just run the .streamlitapp it spontaneously trains the ML models and predicts the values and displays them .
