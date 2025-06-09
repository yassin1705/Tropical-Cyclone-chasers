=============
API Reference
=============

This section provides detailed documentation for the AI-Tornado-Chaser API endpoints and usage.

REST API Endpoints
---------------

Trajectory Prediction
~~~~~~~~~~~~~~~~~~

POST /api/v1/predict
^^^^^^^^^^^^^^^^^^^

Predicts future tornado positions based on historical trajectory data.

**Request Format:**

.. code-block:: json

    {
        "coordinates": [
            [latitude1, longitude1],
            [latitude2, longitude2],
            [latitude3, longitude3],
            [latitude4, longitude4],
            [latitude5, longitude5]
        ],
        "horizon": 10
    }

* `coordinates`: List of 5 [latitude, longitude] pairs
* `horizon`: Number of future positions to predict (default: 10)

**Response Format:**

.. code-block:: json

    {
        "predictions": [
            [pred_lat1, pred_lon1],
            [pred_lat2, pred_lon2],
            ...
            [pred_lat10, pred_lon10]
        ],
        "confidence": 0.95,
        "timestamp": "2025-06-09T12:00:00Z"
    }

**Example Usage:**

.. code-block:: python

    import requests
    import json

    api_url = "http://localhost:5000/api/v1/predict"
    data = {
        "coordinates": [
            [57, 1399],
            [59, 1393],
            [63, 1387],
            [67, 1380],
            [72, 1373]
        ],
        "horizon": 10
    }

    response = requests.post(api_url, json=data)
    predictions = response.json()

Chatbot Interaction
~~~~~~~~~~~~~~~~

POST /api/v1/chat
^^^^^^^^^^^^^^^

Provides intelligent responses to user queries about tornado safety and information.

**Request Format:**

.. code-block:: json

    {
        "query": "What should I do during a tornado warning?",
        "context": {
            "location": [latitude, longitude],
            "current_warning": true
        }
    }

**Response Format:**

.. code-block:: json

    {
        "response": "Find shelter immediately...",
        "confidence": 0.98,
        "sources": ["safety_guidelines", "emergency_protocol"],
        "timestamp": "2025-06-09T12:00:00Z"
    }

Error Handling
-----------

The API uses standard HTTP status codes:

* 200: Successful request
* 400: Bad request (invalid input)
* 404: Resource not found
* 500: Server error

Error responses include detailed messages:

.. code-block:: json

    {
        "error": "Invalid input format",
        "detail": "Coordinates must be a list of 5 lat-long pairs",
        "code": "INVALID_INPUT"
    }

Rate Limiting
----------

API endpoints are rate-limited:

* Prediction API: 100 requests per minute
* Chatbot API: 60 requests per minute

Rate limit headers are included in responses:

.. code-block:: text

    X-RateLimit-Limit: 100
    X-RateLimit-Remaining: 95
    X-RateLimit-Reset: 1623240000

Authentication
------------

API access requires an API key passed in the header:

.. code-block:: python

    headers = {
        "Authorization": "Bearer your_api_key_here",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, 
                           headers=headers,
                           json=data)

Websocket API
-----------

For real-time updates, a WebSocket API is available:

.. code-block:: python

    import websockets
    import asyncio

    async def get_updates():
        uri = "ws://localhost:5000/ws/updates"
        async with websockets.connect(uri) as websocket:
            while True:
                update = await websocket.recv()
                print(json.loads(update))

Model Versioning
-------------

The API supports multiple model versions:

* Specify version in URL: `/api/v1/predict`
* Use header: `X-Model-Version: 2.0`
* Default to latest stable version

Data Formats
----------

All API endpoints:

* Accept and return JSON
* Use UTC timestamps
* Support GZIP compression
* Handle CORS for web clients
