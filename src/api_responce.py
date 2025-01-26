from flask import jsonify

def api_response( success=True, message='', response_code=200,data=None):
    response = {
        "success": success,
        "message": message,
        "response_code" : response_code,
        "data": data
    }
    return jsonify(response), response_code