from flask import Flask, request, jsonify
import os
from Predict import Main
import glob
from Predict_color import MainFunction
import sqlite3
from Suggestion import main
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def db_connect():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    return conn, cursor

def db_disconnect(conn, cursor):
    if conn and cursor:
        cursor.close()
        conn.close()


@app.route('/login', methods=['POST'])
def login():
    data = request.form
    if 'user_id' not in data or 'password' not in data:
        return jsonify({'error': 'Missing user_id or password'}), 400

    user_id = data['user_id']
    password = data['password']

    conn, cursor = db_connect()
    cursor.execute("SELECT * FROM User WHERE Id = ? AND Password = ?", (user_id, password))
    row = cursor.fetchone()
    db_disconnect(conn, cursor)

    if row:
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/signup', methods=['POST'])
def signup():
    data = request.form
    if 'user_id' not in data or 'password' not in data or 'name' not in data:
        return jsonify({'error': 'Missing user_id, name or password'}), 400

    user_id  = data['user_id']
    name     = data['name']
    password = data['password']
    contact  = data.get('contact', '')

    conn, cursor = db_connect()
    cursor.execute("SELECT * FROM User WHERE Id = ?", (user_id,))
    existing = cursor.fetchone()
    if existing:
        db_disconnect(conn, cursor)
        return jsonify({'message': 'User with this user_id already exists'}), 409

    try:
        cursor.execute(
            "INSERT INTO User (Id, Name, Password, Contact) VALUES (?, ?, ?, ?)",
            (user_id, name, password, contact)
        )
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500
    finally:
        db_disconnect(conn, cursor)


@app.route('/Predict', methods=['POST'])
def predict():
    try:
        if 'Image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        if 'user_id' not in request.form:
            return jsonify({'error': 'Missing user_id'}), 400

        user_id    = request.form['user_id']
        image_data = request.files['Image']

        # Clear old images and save new one
        for f in glob.glob('images/*'):
            os.remove(f)
        image_path = f'images/{image_data.filename}'
        image_data.save(image_path)

        # Run prediction
        predicted_type  = Main(image_path)
        predicted_color = MainFunction(image_path)

        # Save image path to Images table
        conn, cursor = db_connect()
        cursor.execute("INSERT INTO Images (Image_Path) VALUES (?)", (image_path,))
        image_id = cursor.lastrowid

        # Save prediction to Suggestions table
        cursor.execute(
            "INSERT INTO Suggestions (User_id, Image_id, Predicted_color, Predicted_type) VALUES (?, ?, ?, ?)",
            (user_id, image_id, predicted_color, predicted_type)
        )
        conn.commit()
        db_disconnect(conn, cursor)

        return jsonify({'Type': predicted_type, 'Color': predicted_color}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/Suggestion', methods=['POST'])
def suggestion():
    try:
        data    = request.form
        if 'user_id' not in data:
            return jsonify({'error': 'Missing user_id'}), 400
        user_id = data['user_id']
        res     = main(user_id)
        return jsonify({'result': res}), 200
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3377, debug=True)
