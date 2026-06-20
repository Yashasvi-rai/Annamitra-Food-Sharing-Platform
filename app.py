import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

from flask import Flask, render_template, request, jsonify, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

# 🔐 Secure Session Configuration (ADD HERE)
from datetime import timedelta

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # change to True when using HTTPS
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ---------------- Database ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Donation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    donor_name = db.Column(db.String(100))
    food_name = db.Column(db.String(100))
    food_type = db.Column(db.String(20))
    quantity = db.Column(db.Integer)
    location = db.Column(db.String(100))
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- ML Setup ----------------
model = tf.keras.models.load_model(
    "model/food_classifier_model1.h5",
    compile=False
)

# ---------------- Veg / Non-Veg Mapping ----------------
DATASET_ROOT = "dataset_flat"

food_to_category = {}

for main_class in os.listdir(DATASET_ROOT):
    main_path = os.path.join(DATASET_ROOT, main_class)

    if os.path.isdir(main_path):
        for sub_class in os.listdir(main_path):
            if os.path.isdir(os.path.join(main_path, sub_class)):
                if main_class.lower() == "veg":
                    food_to_category[sub_class.lower()] = "Veg"
                elif main_class.lower() == "nonveg":
                    food_to_category[sub_class.lower()] = "Non-Veg"

with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return classes[np.argmax(pred)]

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template("home.html")

# ---------------- Signup ----------------
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = bcrypt.generate_password_hash(
            request.form.get("password")).decode('utf-8')

        if User.query.filter_by(email=email).first():
            return render_template("signup.html")

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")

    return render_template("signup.html")

# ---------------- Login (AJAX JSON) ----------------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user'] = email
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "Invalid credentials"})

    return render_template("login.html")

# ---------------- Dashboard ----------------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect("/login")
    return render_template("dashboard.html")

# ---------------- Serve Food (Corrected) ----------------
@app.route('/serve_food', methods=['POST'])
def serve_food():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    # Get form data
    file = request.files.get('file')
    donor_name = request.form.get("donor_name")
    quantity = request.form.get("quantity")
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded image
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # Predict food name using ML model
    food_name = predict_food(path)

    # Get Veg / Non-Veg from dataset mapping
    food_type = food_to_category.get(food_name.lower(), "Unknown")

    # Save to database
    donation = Donation(
        donor_name=donor_name,
        food_name=food_name,
        food_type=food_type,
        quantity=int(quantity) if quantity else 0,
        location="Current Location",
        lat=float(lat) if lat else None,
        lon=float(lon) if lon else None
    )

    db.session.add(donation)
    db.session.commit()

    # Return response to frontend
    return jsonify({
        "food_name": food_name,
        "food_type": food_type,
        "quantity": quantity,
        "donor_name": donor_name,
        "location": "Current Location"
    })

# ---------------- Search (Matches search.html) ----------------
@app.route('/search')
def search():
    item = request.args.get("item")
    donations = Donation.query.filter(
        Donation.food_name.ilike(f"%{item}%")
    ).all()

    results = []
    for d in donations:
        results.append({
            "food_name": d.food_name,
            "food_type": d.food_type,
            "quantity": d.quantity,
            "donor_name": d.donor_name,
            "location": d.location,
            "lat": d.lat,
            "lon": d.lon,
            "distance_km": 1.2
        })

    return jsonify(results)
# ---------------- Run ----------------
if __name__ == "__main__":
    with app.app_context():
        db.drop_all()
        db.create_all()

    app.run(host="0.0.0.0", port=5000, debug=True)




