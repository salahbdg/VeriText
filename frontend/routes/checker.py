from flask import Blueprint, render_template, request, jsonify
import requests

checker_bp = Blueprint('checker', __name__)

@checker_bp.route('/check')
def check():
    return render_template('check.html')

@checker_bp.route('/check_text', methods=['POST'])
def check_text():
    data = request.get_json()
    user_text = data.get('text')
    model_name = data.get('model', 'roberta')  # Default to 'roberta'

    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    # Call the server API
    url = "http://localhost:8080/api/check"
    response = requests.get(url, params={"query": user_text, "model": model_name})

    if response.status_code != 200:
        return jsonify({'error': 'Failed to connect to the server'}), 500

    # Forward all data from the server response to the frontend
    return jsonify(response.json())