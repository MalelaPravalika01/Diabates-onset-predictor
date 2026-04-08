import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Enter details:")

preg = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("DPF: "))
age = float(input("Age: "))

data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

result = model.predict(data)

if result[0] == 1:
    print("⚠️ High Risk of Diabetes")
else:
    print("✅ No Diabetes Risk")