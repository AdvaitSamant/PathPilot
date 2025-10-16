import requests

# ============================================================
# OLLAMA CONNECTION TESTER
# ============================================================

def test_ollama_connection():
    """Check if Ollama is installed and running properly."""
    diagnostics = {
        'api_reachable': False,
        'models_available': [],
        'error_message': None,
        'library_available': False
    }

    # Step 1: Check if ollama library is available
    try:
        import ollama
        diagnostics['library_available'] = True
    except ImportError:
        diagnostics['error_message'] = "Ollama library not installed. Run: pip install ollama"
        return diagnostics

    # Step 2: Check API reachability
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            diagnostics['api_reachable'] = True
            data = response.json()
            diagnostics['models_available'] = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
        else:
            diagnostics['error_message'] = f"Ollama API returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        diagnostics['error_message'] = "Cannot connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        diagnostics['error_message'] = "Ollama connection timeout. Try restarting Ollama."
    except Exception as e:
        diagnostics['error_message'] = f"Unexpected error: {str(e)}"

    return diagnostics


# ============================================================
# OLLAMA RESPONSE METHODS
# ============================================================

def get_ollama_response_method1(context):
    """Method 1: Direct chat using ollama library."""
    try:
        import ollama
        response = ollama.chat(
            model='phi3',
            messages=[
                {"role": "system", "content": "You are a career counselor. Be concise and helpful."},
                {"role": "user", "content": context}
            ],
            options={"temperature": 0.7}
        )
        return True, response['message']['content']
    except Exception as e:
        return False, f"Library method failed: {str(e)}"


def get_ollama_response_method2(context):
    """Method 2: Direct API call to /api/chat."""
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                "model": "phi3",
                "messages": [
                    {"role": "system", "content": "You are a career counselor. Be concise."},
                    {"role": "user", "content": context}
                ],
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json()['message']['content']
        else:
            return False, f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"API method failed: {str(e)}"


def get_ollama_response_method3(context):
    """Method 3: Using /api/generate endpoint."""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "phi3",
                "prompt": f"You are a career counselor. {context}\n\nProvide helpful advice:",
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json().get('response', '')
        else:
            return False, f"Generate API error: {response.status_code}"
    except Exception as e:
        return False, f"Generate method failed: {str(e)}"


# ============================================================
# MASTER HANDLER
# ============================================================

def get_ollama_response(context):
    """Master handler that tries all methods in sequence."""
    methods = [
        ("Ollama Library", get_ollama_response_method1),
        ("Ollama API", get_ollama_response_method2),
        ("Ollama Generate", get_ollama_response_method3)
    ]

    for method_name, method_func in methods:
        success, response = method_func(context)
        if success:
            return response, f"🤖 {method_name}"

    return None, "❌ All Ollama methods failed — fallback to rule-based AI"
