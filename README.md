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

