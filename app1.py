from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# -------------------- Flask Setup --------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annnamitra.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------- ML Model --------------------
MODEL_PATH = r"C:\\Users\\yasha\\Desktop\\Annnamitra\\food_classifier_model1.h5"
JSON_PATH = r"C:\\Users\\yasha\\Desktop\\Annnamitra\\class_indices.json"
DATASET_ROOT = r"C:\\Users\\yasha\\Desktop\\New folder"

model = tf.keras.models.load_model(MODEL_PATH)
with open(JSON_PATH, "r") as f:
    class_indices = json.load(f)
classes = {v: k for k, v in class_indices.items()}

# Map food -> Veg/Non-Veg
food_to_category = {}
for main_class in os.listdir(DATASET_ROOT):
    folder_path = os.path.join(DATASET_ROOT, main_class)
    if not os.path.isdir(folder_path):
        continue
    normalized_name = main_class.lower().replace(" ", "")
    category_label = None
    if normalized_name == "veg":
        category_label = "Veg"
    elif normalized_name == "nonveg":
        category_label = "Non-Veg"
    if category_label:
        for sub in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, sub)):
                food_to_category[sub.lower()] = category_label

# -------------------- Database Models --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(120))

class Food(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    donor_name = db.Column(db.String(120))
    food_name = db.Column(db.String(120))
    food_type = db.Column(db.String(20))
    quantity = db.Column(db.Integer)
    location = db.Column(db.String(200))
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    timestamp = db.Column(db.String(100), default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

# Create tables
with app.app_context():
    db.create_all()

# -------------------- Helper Functions --------------------
def predict_food(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class_name = classes[np.argmax(pred)]
    category = food_to_category.get(pred_class_name.lower(), "Unknown")
    return pred_class_name, category

def haversine(lon1, lat1, lon2, lat2):
    # Calculate distance in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a)) 
    km = 6371 * c
    return km

print("DB absolute path:", os.path.abspath("C:\\Users\\yasha\\Desktop\\Annnamitra\\foodapp\\instance\\annnamitra.db"))

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("home.html")  # Homepage with SignUp/Login

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(email=email).first():
            return jsonify({"success": False, "message": "Email already exists!"})
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        # return jsonify({"success": True, "message": "Account created successfully!"})
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return jsonify({"success": True, "message": "Login successful!"})
        return jsonify({"success": False, "message": "Invalid credentials!"})
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")  # New page: Serve Food / Search Food

# -------------------- Updated Serve Food --------------------
@app.route("/serve_food", methods=["POST"])
def serve_food():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required!"})
    
    file = request.files["file"]
    donor_name = request.form["donor_name"]
    quantity = request.form["quantity"]
    lat = float(request.form["lat"])
    lon = float(request.form["lon"])
    
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)
    
    food_name, food_type = predict_food(save_path)  # food_type = "Veg"/"Non-Veg"
    
    location_text = f"Lat:{lat}, Lon:{lon}"

    new_food = Food(
        donor_name=donor_name,
        food_name=food_name,
        food_type=food_type,
        quantity=int(quantity),
        location=location_text,
        lat=lat,
        lon=lon
    )
    db.session.add(new_food)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Successfully predicted {food_name} ({food_type})!",
        "food_name": food_name,
        "food_type": food_type,  # "Veg" or "Non-Veg"
        "quantity": quantity,
        "donor_name": donor_name,
        "location": location_text
    })


@app.route("/search_food", methods=["GET"])
def search_food():
    if "user_id" not in session:
        return jsonify([])

    item = request.args.get("item")
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    foods = Food.query.filter(Food.food_name.ilike(f"%{item}%")).all()
    result = []
    for f in foods:
        distance = haversine(lon, lat, f.lon, f.lat)
        result.append({
            "food_name": f.food_name,
            "food_type": f.food_type,
            "quantity": f.quantity,
            "donor_name": f.donor_name,
            "location": f.location,
            "lat": f.lat,
            "lon": f.lon,
            "distance_km": round(distance,2),
            "timestamp": f.timestamp
        })
    result.sort(key=lambda x: x["distance_km"])
    return jsonify(result)

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("home"))

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
