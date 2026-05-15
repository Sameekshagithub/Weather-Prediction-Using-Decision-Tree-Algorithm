from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# ── Dataset ──────────────────────────────────────────────────────────
DATA = {
    'Outlook':     ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast',
                    'Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy',
                    'Sunny','Overcast','Rainy','Sunny','Overcast','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool',
                    'Mild','Mild','Mild','Hot','Mild','Mild','Hot','Cool','Hot',
                    'Mild','Hot'],
    'Humidity':    ['High','High','High','High','Normal','Normal','Normal','High',
                    'Normal','Normal','Normal','High','Normal','High','High','Normal',
                    'High','High','High','Normal'],
    'Wind':        ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
                    'Weak','Weak','Strong','Strong','Weak','Strong','Weak','Weak',
                    'Weak','Strong','Strong','Weak'],
    'Play':        ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes',
                    'Yes','Yes','Yes','No','Yes','Yes','No','No','Yes','Yes']
}

FEATURES = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# ── Build & train model once at startup ──────────────────────────────
df = pd.DataFrame(DATA)
le_dict = {}
df_enc = df.copy()
for col in df.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df_enc[FEATURES]
y = df_enc['Play']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
ACC     = round(accuracy_score(y_test, y_pred) * 100, 1)
TRAIN_ACC = round(accuracy_score(y_train, model.predict(X_train)) * 100, 1)
CM      = confusion_matrix(y_test, y_pred).tolist()
CR_DICT = classification_report(y_test, y_pred,
            target_names=le_dict['Play'].classes_, output_dict=True)
IMPORTANCES = dict(zip(FEATURES, [round(float(v), 4) for v in model.feature_importances_]))


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    dataset = df.to_dict(orient='records')
    return render_template('index.html',
        dataset=dataset,
        accuracy=ACC,
        train_accuracy=TRAIN_ACC,
        confusion_matrix=CM,
        classification_report=CR_DICT,
        importances=IMPORTANCES,
        tree_depth=model.get_depth(),
        leaves=model.get_n_leaves(),
        total_records=len(df)
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)
        outlook     = body['outlook']
        temperature = body['temperature']
        humidity    = body['humidity']
        wind        = body['wind']

        encoded = {
            'Outlook':     le_dict['Outlook'].transform([outlook])[0],
            'Temperature': le_dict['Temperature'].transform([temperature])[0],
            'Humidity':    le_dict['Humidity'].transform([humidity])[0],
            'Wind':        le_dict['Wind'].transform([wind])[0],
        }
        sample = pd.DataFrame([encoded])
        pred   = model.predict(sample)[0]
        proba  = model.predict_proba(sample)[0].tolist()
        result = le_dict['Play'].inverse_transform([pred])[0]
        classes = le_dict['Play'].classes_.tolist()

        return jsonify({
            'result': result,
            'confidence': round(max(proba) * 100, 1),
            'probabilities': dict(zip(classes, [round(p*100,1) for p in proba]))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=3000)
