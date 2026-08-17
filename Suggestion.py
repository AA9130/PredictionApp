from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import sqlite3


def db_connect():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    return conn, cursor

def db_disconnect(conn, cursor):
    if conn and cursor:
        cursor.close()
        conn.close()


def main(user_id):
    conn, cursor = db_connect()

    # Get all suggestions
    all_df = pd.read_sql("SELECT * FROM Suggestions", conn)

    # Get the latest prediction for this user
    user_df = pd.read_sql(
        "SELECT * FROM Suggestions WHERE User_id = ? ORDER BY Image_id DESC LIMIT 1",
        conn, params=(user_id,)
    )
    db_disconnect(conn, cursor)

    if user_df.empty:
        return []

    all_df['cloth_features'] = all_df['Predicted_type'] + ' ' + all_df['Predicted_color']

    vectorizer   = TfidfVectorizer()
    cloth_matrix = vectorizer.fit_transform(all_df['cloth_features'])

    cloth_type  = user_df['Predicted_type'].iloc[0]
    cloth_color = user_df['Predicted_color'].iloc[0]

    input_cloth           = cloth_type + ' ' + cloth_color
    input_vectorized      = vectorizer.transform([input_cloth])
    similarity_scores     = linear_kernel(input_vectorized, cloth_matrix).flatten()
    cloth_indices         = similarity_scores.argsort()[::-1]

    user_seen = all_df[all_df['User_id'] == user_id]['cloth_features'].tolist()
    cloth_indices = [idx for idx in cloth_indices if all_df['cloth_features'].iloc[idx] not in user_seen]

    top = all_df.iloc[cloth_indices[:2]][['Predicted_type', 'Predicted_color']].values.tolist()
    return top
