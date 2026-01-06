```python
import jwt
import os
from functools import wraps
from flask import request, jsonify, g

"""
Environment variable for JWT secret key.
"""
# Get the JWT secret key from the environment variable
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

def jwt_required(fn):
    """
    Decorator to require a valid JWT token in the Authorization header.

    Args:
        fn (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to handle JWT token validation.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments.

        Returns:
            response: The response from the decorated function or an error response.
        """
        # Get the Authorization header from the request
        auth_header = request.headers.get("Authorization")

        # Check if the Authorization header is missing
        if not auth_header:
            # Return an error response if the header is missing
            return jsonify({"error": "Authorization header missing"}), 401

        # Split the Authorization header into parts
        parts = auth_header.split()
        # Check if the header is in the correct format (Bearer <token>)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            # Return an error response if the header is invalid
            return jsonify({"error": "Invalid authorization header"}), 401

        # Get the token from the Authorization header
        token = parts[1]

        try:
            # Decode the token using the JWT secret key
            decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            # Store the email globally for this request
            g.email = decoded.get('email')

        except jwt.ExpiredSignatureError:
            # Return an error response if the token has expired
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            # Return an error response if the token is invalid
            return jsonify({"error": "Invalid token"}), 401

        # Call the decorated function
        return fn(*args, **kwargs)

    return wrapper
```