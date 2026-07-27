from flask import Flask, jsonify
import pymssql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd


def db_connect():
    conn = pymssql.connect(server='127.0.0.1', user='sa', password='admin@123', database='master')
    cursor = conn.cursor()
    return conn, cursor

def db_disconnect(conn, cursor):
    if (conn and cursor):
        cursor.close()
        conn.close()


def main(user_id):
    conn, cursor = db_connect()
    q = f"""select * from Result"""
    df = pd.read_sql(q, conn)
    db_disconnect(conn, cursor)
    df['cloth_features'] = df['cloth_types'] + ' ' + df['cloth_colors']
    vectorizer = TfidfVectorizer()

    cloth_matrix = vectorizer.fit_transform(df['cloth_features'])

    cosine_sim = linear_kernel(cloth_matrix, cloth_matrix)

    def get_cloth_recommendations(user_id, cloth_type, cloth_color, num_recommendations=2):
        input_cloth = cloth_type + ' ' + cloth_color

        input_cloth_vectorized = vectorizer.transform([input_cloth])

        similarity_scores = linear_kernel(input_cloth_vectorized, cloth_matrix).flatten()

        cloth_indices = similarity_scores.argsort()[::-1]

        user_liked_cloths = df[df['user_ids'] == user_id]['cloth_features'].tolist()
        cloth_indices = [idx for idx in cloth_indices if df['cloth_features'][idx] not in user_liked_cloths]

        top_recommendations = df.iloc[cloth_indices[:num_recommendations]][['cloth_types', 'cloth_colors']].values.tolist()

        return top_recommendations
    
    conn, cursor = db_connect()
    q = f"""select top 1 * from Result  where user_id = {user_id}"""
    df = pd.read_sql(q, conn)
    db_disconnect(conn, cursor)
    cloth_type = df['Predicted_type'][0]
    cloth_color = df['Predicted_color'][0]
    recommendations = get_cloth_recommendations(user_id, cloth_type, cloth_color)
    return recommendations



