from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from groq import Groq
from dotenv import load_dotenv
import os
import logging

# Ρύθμιση logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

# Ρύθμιση session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Φόρτωση Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY λείπει από το .env!")

client = Groq(api_key=GROQ_API_KEY)
logger.info("Groq client ξεκίνησε επιτυχώς")

# Dictionary για συνομιλίες ανά session
conversations = {}

@app.route('/')
def index():
    logger.debug("Φόρτωση index.html")
    # Δημιουργία session ID αν δεν υπάρχει
    if 'session_id' not in session:
        session['session_id'] = str(hash(request.remote_addr + request.user_agent.string))
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message', '')
    logger.debug(f"Λήψη μηνύματος: {user_message}")

    if not user_message:
        logger.warning("Κενό μήνυμα από τον χρήστη")
        return jsonify({"response": "Παρακαλώ γράψε κάτι πρώτα."})

    session_id = session.get('session_id')
    if session_id not in conversations:
        conversations[session_id] = []
    conversation = conversations[session_id]

    conversation.append({"role": "user", "content": user_message})
    logger.debug(f"Τρέχουσα συνομιλία: {conversation}")

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Χρησιμοποιώ το επιτυχημένο μοντέλο
            messages=conversation,
            max_tokens=500,  # Περιορισμός μήκους
            temperature=0.7  # Δημιουργικότητα
        )

        bot_reply = completion.choices[0].message.content
        logger.info(f"Απάντηση από Groq: {bot_reply}")
        conversation.append({"role": "assistant", "content": bot_reply})
        return jsonify({"response": bot_reply})

    except Exception as e:
        logger.error(f"Σφάλμα κατά την κλήση Groq: {e}")
        return jsonify({"response": f"Σφάλμα: {str(e)}. Δοκίμασε ξανά."})

if __name__ == '__main__':
    logger.info("Server ξεκινά...")
    app.run(debug=True, host='0.0.0.0', port=5000)