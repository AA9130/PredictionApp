from flask import Flask, request, jsonify
import os
from Predict import Main
import glob
from Predict_color import MainFunction
import pymssql
import pandas as pd
from Suggestion import main
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def db_connect():
    conn = pymssql.connect(server='127.0.0.1', user='sa', password='admin@123', database='master')
    cursor = conn.cursor()
    return conn, cursor

def db_disconnect(conn, cursor):
    if (conn and cursor):
        cursor.close()
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    print(data)
    if 'user_id' not in data or 'password' not in data:
        return jsonify({'error': 'Missing user_id or password'}), 400
    user_id = data['user_id']
    password = data['password']

    q = f"select * from users where user_id = '{user_id}' AND password = '{password}'"
    conn, cursor = db_connect()
    df = pd.read_sql(q, conn)
    db_disconnect(conn, cursor)
    if len(df) > 0:
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/signup', methods=['POST'])
def signup():
    data = request.form

    if 'user_id' not in data or 'password' not in data or 'email' not in data:
        return jsonify({'error': 'Missing user_id or password'}), 400

    user_id = data['user_id']
    email = data['email']
    password = data['password']
    contact = data['contact']

    q = f"select * from users where user_id = '{user_id}' AND password = '{password}'"
    conn, cursor = db_connect()
    df = pd.read_sql(q, conn)
    db_disconnect(conn, cursor)
    if len(df) > 0:
        return jsonify({'message': 'User with this user_id already exists'}), 409
    try:
        conn, cursor = db_connect()
        cursor.execute(f"INSERT INTO users (User_id, Email, Password, Contact) VALUES ('{user_id}', '{email}', '{password}', '{contact}')")
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except pymssql.Error as e:
        if conn and cursor:
            db_disconnect(conn, cursor)
        return jsonify({'error': f'Error: {str(e)}'}), 500
    finally:
        db_disconnect(conn, cursor)

@app.route('/Predict', methods=['POST'])
def index():
    try:
        for f in glob.glob('images'):
            os.remove(f)
        image_data = request.files['Image']
        image_data.save(f'images/{image_data.filename}')
        img = glob.glob('images/*')[0]
        Type = Main(img)
        Color = MainFunction(img)
        return {'Type': Type, 'Color':Color}
    except Exception as e:
        return {'error': str(e)}
    

@app.route('/Suggestion', methods=['POST'])
def Suggestion():
    try:
        data = request.form
        user_id = data['user_id']
        res = Main(user_id)
        return jsonify({'result':  res}), 200
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3377, debug=True)