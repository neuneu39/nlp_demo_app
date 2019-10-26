from flask import Flask, request, jsonify
from main import app
from main.nlp_service import NlpService

@app.route("/post", methods=['POST'])
def post():
    json = request.get_json()  # POSTされたJSONを取得
    if 'User_Answer' not in json:
        return jsonify('format is not correct')
    answer = json['User_Answer']

    response_answer = NlpService()
    return jsonify(response_answer.create_response_answer(answer))
