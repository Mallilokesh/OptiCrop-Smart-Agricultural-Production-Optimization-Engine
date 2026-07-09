# OptiCrop

OptiCrop is a Flask-based crop recommendation system that uses soil and environmental measurements to suggest a suitable crop.

## Features

- Trains and compares KNN, Logistic Regression, Decision Tree, Random Forest, and a K-Means baseline.
- Uses Nitrogen, Phosphorous, Potassium, temperature, humidity, pH, and rainfall as model inputs.
- Saves the best trained model as a Pickle file for reuse by the Flask app.
- Provides a responsive Bootstrap interface for instant crop recommendations.
- Includes validation and preprocessing so inputs are checked before prediction.

## Project Structure

```text
OptiCrop/
├── app.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── validation.py
├── scripts/
│   └── train_model.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── css/
│       └── styles.css
├── data/
│   └── .gitkeep
├── artifacts/
│   └── .gitkeep
└── notebooks/
    └── OptiCrop_Model_Training.ipynb
```

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your machine uses the Windows Python launcher:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### PyCharm Setup

If PyCharm already has all required packages installed, open this folder in PyCharm and select that interpreter:

1. Go to `File > Settings > Project > Python Interpreter`.
2. Choose the interpreter or virtual environment that already contains Flask, scikit-learn, pandas, NumPy, matplotlib, seaborn, and Jupyter.
3. Open PyCharm's terminal and run the training and Flask commands below.

## Train The Model

```powershell
python scripts/train_model.py
```

The script creates `data/crop_recommendation_sample.csv` if no dataset exists, evaluates all configured algorithms, and saves the best model to `artifacts/best_model.pkl`.

To use your own dataset, save it as `data/crop_recommendation.csv` with these columns:

```text
N,P,K,temperature,humidity,ph,rainfall,label
```

## Run The Web App

```powershell
flask --app app run --debug
```

Open the local Flask URL in your browser and enter the soil and weather values to receive a crop recommendation.

## Version Control

```powershell
git init
git add .
git commit -m "Initial OptiCrop project"
```
