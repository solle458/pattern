from flask import jsonify
import logging

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """エラーハンドラーの登録"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """400 Bad Request エラーハンドラー"""
        logger.error(f"Bad Request: {str(error)}")
        return jsonify({
            "error": "Bad Request",
            "message": str(error)
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        """404 Not Found エラーハンドラー"""
        logger.error(f"Not Found: {str(error)}")
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """405 Method Not Allowed エラーハンドラー"""
        logger.error(f"Method Not Allowed: {str(error)}")
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The method is not allowed for the requested URL"
        }), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        """500 Internal Server Error エラーハンドラー"""
        logger.error(f"Internal Server Error: {str(error)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "An internal server error occurred"
        }), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """未処理の例外のハンドラー"""
        logger.error(f"Unhandled Exception: {str(error)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }), 500 
