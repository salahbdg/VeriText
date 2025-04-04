from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
def home():
    return render_template('index.html')


@home_bp.route('/explanation')
def what():
    return render_template('explanation.html')