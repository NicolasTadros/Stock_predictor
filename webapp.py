from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    
    data = [
        {"date": "2025-05-10", "actual_price": 150, "predicted_price": 152},
        {"date": "2025-05-11", "actual_price": 153, "predicted_price": 154},
        {"date": "2025-05-12", "actual_price": 155, "predicted_price": 156},
        {"date": "2025-05-13", "actual_price": 157, "predicted_price": 158},
        {"date": "2025-05-14", "actual_price": 160, "predicted_price": 159}
    ]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
