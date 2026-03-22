from flask import Flask, request, jsonify, send_from_directory, make_response, send_file
from Veronica import get_veronica_response, load_knowledge_base, save_knowledge_base, get_gemini_response
from flask_cors import CORS
import os
import google.generativeai as genai
import re
import logging
import uuid
from datetime import datetime
import psycopg2
import psycopg2.extras
import io
import csv

# ---- Configuration ----
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Allowed origins (exact matches)
ALLOWED_ORIGINS = {
    "https://christjuniorcollege.in",
    "https://www.christjuniorcollege.in",
    "https://byncai.net",
    "https://www.byncai.net"
}

# Enable CORS (helps for simple cases; explicit OPTIONS handling below is the key)
CORS(app, resources={r"/predict": {"origins": list(ALLOWED_ORIGINS)}}, methods=["POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Load knowledge base
knowledge_base = load_knowledge_base('knowledge_base.json')

# Configure the Gemini model from environment variable (do not hardcode keys)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found in environment variables"
    )

genai.configure(api_key=api_key)

# Optional: helper to call Gemini if needed (kept from your code)
DATABASE_URL = os.environ.get('DATABASE_URL')  # expected to be provided in environment

if not DATABASE_URL:
    app.logger.warning('DATABASE_URL environment variable not set. DB logging and export will be disabled.')


def get_db_conn():
    """Return a new psycopg2 connection using DATABASE_URL. Caller should close the connection."""
    if not DATABASE_URL:
        raise RuntimeError('DATABASE_URL not configured')
    return psycopg2.connect(DATABASE_URL, sslmode=os.environ.get('PGSSLMODE', 'prefer'))


def init_db():
    """Create messages table if it does not exist."""
    if not DATABASE_URL:
        return
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS veronica_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    message TEXT,
                    reply_id TEXT,
                    url TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """)
                conn.commit()
        app.logger.info('Initialized veronica_messages table')
    except Exception:
        app.logger.exception('Failed to initialize DB')


# helper to insert a message row
def db_insert_message(session_id, role, message, reply_id=None, url=None, user_agent=None, created_at=None):
    if not DATABASE_URL:
        return
    try:
        created_at = created_at or datetime.utcnow()
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO veronica_messages (session_id, role, message, reply_id, url, user_agent, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (session_id, role, message, reply_id, url, user_agent, created_at)
                )
                conn.commit()
    except Exception:
        app.logger.exception('db_insert_message failed')


# helper to fetch all conversations (ordered by created_at)
def fetch_all_conversations():
    if not DATABASE_URL:
        return []
    try:
        with get_db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT session_id, role, message, reply_id, url, user_agent, created_at FROM veronica_messages ORDER BY created_at ASC")
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    except Exception:
        app.logger.exception('fetch_all_conversations failed')
        return []


# Initialize DB (create table)
init_db()

# Optional: helper to call Gemini if needed (kept from your code)
def get_veronica_response_from_knowledge_or_gemini(text):
    resp = get_veronica_response(text, knowledge_base)
    if resp == "Sorry I dont know what you are talking about! ^.^":
        resp = get_gemini_response(text)
    return resp

# DEBUG: log incoming requests (helps see if preflight reaches Flask)
@app.before_request
def log_request():
    app.logger.debug("Incoming %s %s", request.method, request.url)
    app.logger.debug("Headers: %s", dict(request.headers))

# /predict route: explicit OPTIONS + POST handling
@app.route("/predict", methods=["OPTIONS", "POST"], strict_slashes=False)
def predict():
    # Preflight: securely respond with CORS headers if origin trusted
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "")
        resp = make_response("", 204)
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

    try:
        request_data = request.get_json() or {}
        text = request_data.get("message", "")
        # session id may be sent by widget (optional)
        session_id = request_data.get('session_id') or request_data.get('sid') or 'unknown'
        url = request_data.get('url')
        
        user_agent = request_data.get('user_agent') or request.headers.get('User-Agent')

        if not text:
            return jsonify({"answer": "Invalid input"}), 400

        user_text = text.strip().lower()

        # 1) Detect full URL
        match = re.search(r"(https?://[^\s]+)", user_text)
        if match:
            url_match = match.group(0)
            # log user message and bot reply indicating opening the link
            try:
                db_insert_message(session_id, 'user', text, reply_id=None, url=url or url_match, user_agent=user_agent)
                # create a bot reply id
                bot_reply_id = str(uuid.uuid4())[:12]
                db_insert_message(session_id, 'bot', f"Opening the link: {url_match}...", reply_id=bot_reply_id, url=url or url_match, user_agent=user_agent)
            except Exception:
                app.logger.exception('logging url branch failed')
            return jsonify({"answer": f"Opening the link: {url_match}...", "url": url_match}), 200

        # --- CJC MAPPING ---
        cj_base = "https://christjuniorcollege.in/"
        mapping = {
            "institution": ("Open Institution", cj_base + "about-the-institution.php"),
            "emblem": ("Open Emblem", cj_base + "about-the-emblem.php"),
            "campus culture": ("Open Campus Culture", cj_base + "about-campus-culture.php"),
            "founder": ("Open Founder", cj_base + "about-founder.php"),
            "vision mission": ("Open Vision & Mission", cj_base + "about-vision-mission.php"),
            "principal": ("Open Principal's Message", cj_base + "about-principals-message.php"),
            "cmi": ("Open CMI Education Policy", cj_base + "about-cmi-and-cmi-education-policy.php"),
            "campus": ("Open Campus", cj_base + "about-campus.php"),
            "college profile": ("Open College Profile", cj_base + "about-college-profile.php"),
            "academic growth": ("Open Academic Growth", cj_base + "about-academic-growth.php"),
            "educational policies": ("Open Evolution of Educational Policies", cj_base + "about-evolution-of-educational-policies.php"),
            "student development": ("Open Student Development", cj_base + "about-student-development.php"),
            "expansion": ("Open Expansion", cj_base + "about-expansion.php"),
            "infrastructure": ("Open Infrastructure", cj_base + "about-infrastructure.php"),
            "pu academics": ("Open PU Academics", cj_base + "academics.php"),
            "faculty": ("Open Faculty", cj_base + "faculty.php"),
            "pu programs": ("Open PU Programs", cj_base + "pre-university-course.php#academic-programs"),
            "admission pu": ("Open PU Admission", cj_base + "admission-pu.php"),
            "enquiry pu": ("Open PU Admission Enquiry", cj_base + "admission-enquiry-pu.php"),
            "pu faqs": ("Open PU FAQs", cj_base + "faqs-pu.php"),
            "student life": ("Open Student Life", cj_base + "student-life.php"),
            "achievers": ("Open Achievers", cj_base + "achievers.php"),
            "pu blog": ("Open PU Blog", "https://christjuniorcollege.wordpress.com/"),
            "ibdp blog": ("Open IBDP Blog", "https://cjcibdp.wordpress.com/"),
            "contact pu": ("Open PU Contact", cj_base + "contact-us-pu.php"),
            "contact ibdp": ("Open IBDP Contact", cj_base + "contact-us-ibdp.php"),
            "about ibdp": ("Open IBDP About", cj_base + "about-us-ibdp.php"),
            "ibdp programs": ("Open IBDP Programs", cj_base + "academics-programs-ibdp.php"),
            "admission ibdp": ("Open IBDP Admission Enquiry", cj_base + "admission-enquiry-ibdp.php"),
            "ibdp process": ("Open IBDP Admissions Process", cj_base + "admissions-process-ibdp.php"),
            "ibdp faqs": ("Open IBDP FAQs", cj_base + "admission-faqs-ibdp.php"),
            "publications": ("Open IBDP Publications", cj_base + "publications.php"),
            "managebac": ("Open ManageBac Login", "https://cjc.managebac.com/login")
        }

        # --- FIXED COMMAND DETECTION ---
        words = user_text.split()
        if words and words[0] == "open":
            query = " ".join(words[1:]).strip()

            for key, (label, url_map) in mapping.items():
                if key in query:
                    # log user and bot
                    try:
                        db_insert_message(session_id, 'user', text, reply_id=None, url=url, user_agent=user_agent)
                        bot_reply_id = str(uuid.uuid4())[:12]
                        db_insert_message(session_id, 'bot', f"{label}...", reply_id=bot_reply_id, url=url_map, user_agent=user_agent)
                    except Exception:
                        app.logger.exception('logging mapping branch failed')

                    return jsonify({"answer": f"{label}...", "url": url_map}), 200

            # If no match, DO NOTHING — NO GOOGLE SEARCH
            try:
                db_insert_message(session_id, 'user', text, reply_id=None, url=url, user_agent=user_agent)
                bot_reply_id = str(uuid.uuid4())[:12]
                db_insert_message(session_id, 'bot', "I couldn't find a matching command. Try again with clearer words.", reply_id=bot_reply_id, url=url, user_agent=user_agent)
            except Exception:
                app.logger.exception('logging mapping-miss branch failed')

            return jsonify({"answer": "I couldn't find a matching command. Try again with clearer words."}), 200

        # --- AI RESPONSE (session-aware with Redis + Gemini) ---
        response = get_veronica_response(
            user_question=text,
            knowledge_base=knowledge_base,
            session_id=session_id
        )

        # log user message and bot response to DB (non-blocking)
        try:
            db_insert_message(session_id, 'user', text, reply_id=None, url=url, user_agent=user_agent)
            bot_reply_id = str(uuid.uuid4())[:12]
            db_insert_message(session_id, 'bot', response, reply_id=bot_reply_id, url=url, user_agent=user_agent)
        except Exception:
            app.logger.exception('db logging for ai response failed')

        return jsonify({"answer": response}), 200

    except Exception:
        app.logger.exception("Error in /predict")
        return jsonify({"error": "An unexpected error occurred."}), 500



# Endpoint to export all conversations as CSV
@app.route('/export_conversations', methods=['GET'])
def export_conversations():
    try:
        rows = fetch_all_conversations()
        if not rows:
            return jsonify({'ok': True, 'file': None, 'message': 'No conversation rows found or DB not configured.'})

        # create CSV in-memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['session_id', 'role', 'message', 'reply_id', 'url', 'user_agent', 'created_at'])
        for r in rows:
            writer.writerow([
                r.get('session_id'),
                r.get('role'),
                (r.get('message') or '').replace('\n', '\\n'),
                r.get('reply_id'),
                r.get('url'),
                r.get('user_agent'),
                r.get('created_at').isoformat() if r.get('created_at') else ''
            ])

        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='veronica_conversations.csv')
    except Exception:
        app.logger.exception('export_conversations failed')
        return jsonify({'ok': False, 'message': 'Export failed'}), 500


# Static file endpoints (unchanged)
@app.route("/widget.js")
def widget_js():
    return send_from_directory(app.static_folder, 'widget.js')

@app.route("/")
def home():
    return send_from_directory('templates', 'index.html')

@app.route("/founders")
def founder():
    return send_from_directory('templates','founders.html')

@app.route("/cogniai")
def cogniai():
    return send_from_directory('templates','cogniai.html')

@app.route("/about")
def pro():
    return send_from_directory('templates', 'about.html')

@app.route("/services")
def details():
    return send_from_directory('templates', 'services.html')

@app.route("/contact")
def contact():
    return send_from_directory('templates', 'contact.html')

@app.route("/notification")
def getpro():
    return send_from_directory('templates', 'notifications.html')

@app.route("/submit-details", methods=["POST"])
def submit_details():
    return jsonify({"message": "Form submitted successfully!"})

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route("/save", methods=["POST"])
def save():
    global knowledge_base
    new_knowledge_base = request.get_json().get("knowledge_base")
    save_knowledge_base(new_knowledge_base, 'knowledge_base.json')
    knowledge_base = new_knowledge_base
    return jsonify({"message": "Knowledge base saved successfully!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
