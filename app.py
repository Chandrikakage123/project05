import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "supersecretkey"

BASE_DIR = os.path.dirname(__file__)
DATABASE = os.path.join(BASE_DIR, "database.db")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL
    )
    """)

    # default admin
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username,password,role) VALUES (?,?,?)",
            ("admin", "admin123", "admin")
        )

    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(BASE_DIR, "pneumonia_model.h5")
model = tf.keras.models.load_model(model_path, compile=False)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return "PNEUMONIA", float(pred*100)
    else:
        return "NORMAL", float((1-pred)*100)

# ---------------- ACCESS PAGE ----------------
@app.route("/")
def access():
    return render_template("access.html")

# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=? AND role='admin'",
                       (username,password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return "Invalid Credentials"

    return render_template("admin_login.html")

# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db()
    records = conn.execute("SELECT * FROM patients").fetchall()
    conn.close()

    return render_template("admin_dashboard.html", records=records)

# ---------------- DELETE ----------------
@app.route("/delete/<int:id>")
def delete_patient(id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db()
    conn.execute("DELETE FROM patients WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for("admin_dashboard"))

# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        file = request.files["xray"]

        if file.filename == "":
            return "No File Selected"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        

        status, confidence = predict_image(filepath)

        conn = get_db()
        conn.execute("""
        INSERT INTO patients(name,age,gender,image_path,prediction,confidence)
        VALUES (?,?,?,?,?,?)
        """,(name,age,gender,filename,status,confidence))
        conn.commit()
        conn.close()

        return render_template("result.html",
                               status=status,
                               confidence=confidence,
                               image=filename)

    return render_template("upload.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("access"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)