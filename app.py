from flask import Flask, request, jsonify, render_template
from automation_engine import RAGAutomationEngine

app = Flask(__name__)
engine = RAGAutomationEngine()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    query = request.form.get("query", "").strip()
    file = request.files.get("file")

    session_id = "user1"

    if not query:
        return jsonify({
            "answer": "Please enter a question.",
            "attack_detected": False,
            "attack_type": None,
            "score": 0.0
        })

    # upload doc if present
    if file and file.filename:
        upload_result = engine.upload_document(session_id, file.read())

        if upload_result["status"] == "error":
            return jsonify({
                "answer": upload_result["message"],
                "attack_detected": False,
                "attack_type": None,
                "score": 0.0
            })

    result = engine.query(session_id, query)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)