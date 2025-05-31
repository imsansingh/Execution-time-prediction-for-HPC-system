import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
df = pd.read_csv("hpc_resource_prediction_dataset.csv")
X = df.drop(columns=['Y'])
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

template = '''
<!DOCTYPE html>
<html>
<head>
    <title>HPC Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to right, #4facfe, #00f2fe);
            padding: 40px;
            color: #333;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        h2 {
            text-align: center;
        }
        label {
            display: block;
            margin-top: 15px;
        }
        input[type=range] {
            width: 100%;
        }
        span.output {
            font-weight: bold;
            color: #0066cc;
        }
        button {
            margin-top: 25px;
            width: 100%;
            padding: 12px;
            font-size: 16px;
            background-color: #0066cc;
            color: white;
            border: none;
            border-radius: 6px;
        }
        button:hover {
            background-color: #004999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>HPC Execution Time Predictor</h2>
        <form method="post">
            <label>X1: Problem Size <span id="outX1" class="output">50000</span></label>
            <input type="range" name="X1" id="X1" min="1028" max="99991" value="50000" oninput="outX1.innerText = this.value">

            <label>X2: Threads <span id="outX2" class="output">528</span></label>
            <input type="range" name="X2" id="X2" min="32" max="1024" value="528" oninput="outX2.innerText = this.value">

            <label>X3: Blocks <span id="outX3" class="output">16</span></label>
            <input type="range" name="X3" id="X3" min="1" max="32" value="16" oninput="outX3.innerText = this.value">

            <label>X4: Grids <span id="outX4" class="output">16</span></label>
            <input type="range" name="X4" id="X4" min="1" max="32" value="16" oninput="outX4.innerText = this.value">

            <label>X5: CPU+GPU Cores <span id="outX5" class="output">128</span></label>
            <input type="range" name="X5" id="X5" min="16" max="512" value="128" oninput="outX5.innerText = this.value">

            <label>X6: Iterations <span id="outX6" class="output">5000</span></label>
            <input type="range" name="X6" id="X6" min="100" max="10000" value="5000" oninput="outX6.innerText = this.value">

            <label>X7: Clock Rate (GHz) <span id="outX7" class="output">2</span></label>
            <input type="range" name="X7" id="X7" min="1" max="3" step="0.1" value="2" oninput="outX7.innerText = this.value">

            <label>X8: FLOPS <span id="outX8" class="output">1.0e12</span></label>
            <input type="range" name="X8" id="X8" min="100245008771" max="1999736918795" value="1000000000000" step="100000000000" oninput="outX8.innerText = this.value">

            <label>X9: Bandwidth (GB/s) <span id="outX9" class="output">30</span></label>
            <input type="range" name="X9" id="X9" min="10" max="49" value="30" oninput="outX9.innerText = this.value">

            <label>X10: PCI-e Bandwidth <span id="outX10" class="output">16</span></label>
            <input type="range" name="X10" id="X10" min="8" max="31" value="16" oninput="outX10.innerText = this.value">

            <button type="submit">Predict Execution Time</button>
        </form>
        {% if prediction is not none %}
            <h3 style="text-align:center;">Predicted Execution Time: <span class="output">{{ prediction }} seconds</span></h3>
        {% endif %}
    </div>
</body>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        inputs = [float(request.form[f"X{i}"]) for i in range(1, 11)]
        df_input = pd.DataFrame([inputs], columns=[f"X{i}" for i in range(1, 11)])
        prediction = round(rf.predict(df_input)[0], 2)
    return render_template_string(template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
