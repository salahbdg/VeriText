from flask import Flask, render_template
from routes.home import home_bp
from routes.checker import checker_bp

app = Flask(__name__)

# Register Blueprints (Modularized Routes)
app.register_blueprint(home_bp)
app.register_blueprint(checker_bp)

if __name__ == '__main__':
    app.run(debug=True)
