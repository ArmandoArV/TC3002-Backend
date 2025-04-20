from flask import Blueprint, jsonify

healthcheck_bp = Blueprint('healthcheck', __name__)

@healthcheck_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200