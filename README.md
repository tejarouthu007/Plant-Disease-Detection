# 🌱 Plant Disease Detection App

A deep learning-based web app built with **TensorFlow** and **Streamlit** to classify plant diseases from images.

## 🚀 Features
- Upload an image of a plant leaf.
- Get instant disease classification using a pre-trained model.
- Simple and user-friendly Streamlit UI.

## 📂 Project Structure
```
📁 plant-disease-detection/
├── 📂 app/
│   ├── main.py          # Streamlit application
│   ├── class_indices.json # Class labels for predictions
├── 📂 model/
│   ├── plant_disease_prediction_model.keras # Deep learning model (not included in repo)
├── .gitignore           # Ignore virtual env and model files
├── requirements.txt     # Required dependencies
├── runtime.txt          # Python version for Streamlit Cloud
└── README.md            # Project documentation
```

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/tejarouthu007/plant-disease-detection.git
cd plant-disease-detection
```

### **2️⃣ Create and Activate a Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Download the Model**
Since the model is too large for GitHub, download it manually or use the automated download in `main.py`:
```bash
python app/main.py  # This will trigger gdown to fetch the model
```
If needed, manually download from [Google Drive](https://drive.google.com/file/d/1--JDi46vVyMu3KLwnFCcdh78bX8e-YoI/view?usp=sharing) and place it in `trained_model/`.

### **5️⃣ Run the Streamlit App**
```bash
streamlit run app/main.py
```

## 🚀 Deploying to Streamlit Cloud
### **1️⃣ Push Your Code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```
### **2️⃣ Deploy on Streamlit Cloud**
1. Go to [Streamlit Cloud](https://share.streamlit.io/).
2. Click **New app** and connect your GitHub repo.
3. Ensure **requirements.txt** is detected.
4. Click **Deploy** and wait for the app to go live! 🎉

## 🔧 Troubleshooting
- **App crashes due to missing model?** Ensure the model is downloaded and placed in `trained_model/`.
- **Wrong Python version?** Add a `runtime.txt` with `3.10` inside.
- **Other issues?** Check logs in Streamlit Cloud or run locally with `streamlit run app/main.py`.


💡 **Questions or Contributions?** Feel free to open an issue or submit a PR! 🚀

