import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Load pickled models and data
model_reg = pickle.load(open('model_reg.pkl', 'rb'))
model_clf = pickle.load(open('model_clf.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Recommendation System Functions
def recommend_attractions_content_based(attraction_id, df, n=5):
    attraction_features = df[df["AttractionId"] == attraction_id][["AttractionType", "UserId"]].values
    knn = NearestNeighbors(n_neighbors=n, metric="cosine")
    knn.fit(df[["AttractionType", "UserId"]])
    distances, indices = knn.kneighbors(attraction_features)
    recommendations = df.iloc[indices[0]]["Attraction"].tolist()
    return recommendations

def recommend_attractions_collaborative(user_id, df, n=5):
    user_item_matrix = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean")
    user_sparse_matrix = csr_matrix(user_item_matrix.fillna(0))
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(user_sparse_matrix)
    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors(user_sparse_matrix[user_index], n_neighbors=n + 1)
    similar_users = indices.flatten()[1:]
    recommendations = []
    for sim_user in similar_users:
        top_attractions = user_item_matrix.iloc[sim_user].sort_values(ascending=False).index[:n]
        recommendations.extend(top_attractions)
    return list(set(recommendations))[:n]

# Streamlit App
st.title("Tourism Experience Analytics")
st.sidebar.title("🌍 Travel Experience Dashboard")
option = st.sidebar.selectbox("Choose an analysis", ["Prediction", "Classification", "Recommendation", "SQL Insights"])

if option == "Prediction":
    st.subheader("🤖 Rating Prediction")
    cont_id = st.number_input("Continent ID", value=1, step=1)
    country_id = st.number_input("Country ID", value=2, step=1)
    # visit_mode_x = st.number_input("Visit Mode X", value=0, step=1)
    # visit_mode_y = st.number_input("Visit Mode Y", value=10, step=1)
    visit_month = st.number_input("Visit Month", value=7, step=1)
    attraction_type_id = st.number_input("Attraction Type ID", value=63, step=1)

    if st.button("Predict Rating"):
        #pred_rating = model_reg.predict([[cont_id, country_id, visit_mode_x, visit_mode_y, visit_month, attraction_type_id]])[0]
        pred_rating = model_reg.predict([[cont_id, country_id, visit_month, attraction_type_id]])[0]
        #st.write(f"Predicted Rating: {pred_rating}")
        st.write(f"Predicted Rating: {pred_rating}")

elif option == "Classification":
    st.subheader("📊 Visit Mode Classification")
    cont_id_clf = st.number_input("Continent ID (Classification)", value=1, step=1)
    country_id_clf = st.number_input("Country ID (Classification)", value=2, step=1)
    user_visit_count_clf = st.number_input("User Visit Count", value=1, step=1)
    attraction_popularity_clf = st.number_input("Attraction Popularity", value=0.5, step=0.01)

    if st.button("Predict Visit Mode"):
        pred_visit_mode = model_clf.predict([[cont_id_clf, country_id_clf, user_visit_count_clf, attraction_popularity_clf]])[0]
        st.write(f"Predicted Visit Mode: {pred_visit_mode}")

        #Get the Classification report.
        st.subheader("Classification Report")
        X_clf = df[["ContenentId", "CountryId_x", "User_Visit_Count", "Attraction_Popularity"]]
        y_clf = df["VisitMode_y"]
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        y_pred_clf = model_clf.predict(X_test_clf)
        st.text(classification_report(y_test_clf, y_pred_clf))

elif option == "Recommendation":
    st.subheader("🎯 Recommendations")
    recommendation_type = st.sidebar.radio("Choose Recommendation Type", ["Content-Based", "Collaborative Filtering"])

    if recommendation_type == "Content-Based":
        attraction_input = st.number_input("Enter Attraction ID:", min_value=int(df["AttractionId"].min()), max_value=int(df["AttractionId"].max()))
        if st.button("Get Recommendations"):
            st.write(f"Recommended suggestion for attractions: {recommend_attractions_content_based(attraction_input, df)}")

    elif recommendation_type == "Collaborative Filtering":
        user_input = st.number_input("Enter User ID:", min_value=int(df["UserId"].min()), max_value=int(df["UserId"].max()))
        if st.button("Get Recommendations"):
            st.write(f"Recommended attraction_ids for User ID {user_input}: {recommend_attractions_collaborative(user_input, df)}")

elif option == "SQL Insights":
    st.subheader("📌 SQL Insights")
    conn = sqlite3.connect('tourism_db.db')
    cursor = conn.cursor()

    query = """
    SELECT Attraction, ROUND(AVG(Rating),2) AS AvgRating
    FROM merged_dataset
    GROUP BY Attraction
    ORDER BY AvgRating DESC
    LIMIT 10;
    """
    cursor.execute(query)
    st.write(pd.DataFrame(cursor.fetchall()))
    conn.close()