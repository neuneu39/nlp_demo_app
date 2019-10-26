from flask import Flask, request, jsonify
from nlp_service import NlpService
import random
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/post", methods=['POST'])
def post():
    json = request.get_json()  # POSTされたJSONを取得
    if 'User_Answer' not in json:
        return jsonify('format is not correct')
    answer = json['User_Answer']

    response_answer = NlpService(answer)
    return jsonify(response_answer.create_response_answer())


if __name__ == '__main__':
    # http://localhost:5000/ でアクセスできるよう起動
    app.run(host='localhost', port=5000, debug=True)

