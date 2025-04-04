from flask import Blueprint, render_template, request, jsonify

checker_bp = Blueprint('checker', __name__)

@checker_bp.route('/check')
def check():
    return render_template('check.html')

@checker_bp.route('/check_text', methods=['POST'])
def check_text():
    data = request.get_json()
    user_text = data.get('text')

    # You can do actual logic here, but for now we'll simulate a response
    mock_probability = 72.5  # example value

    return jsonify({
        'probability': mock_probability
    })