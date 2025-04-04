from flask import Blueprint, render_template

checker_bp = Blueprint('checker', __name__)

@checker_bp.route('/check')
def check():
    return render_template('check.html')
