"""WSGI entry point.

Production: gunicorn -c gunicorn.conf.py wsgi:app
Local dev:  python wsgi.py
"""

from chatbot import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
