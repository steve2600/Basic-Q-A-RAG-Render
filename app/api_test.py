import requests

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8",
    "Content-Type": "application/json"
}

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "Does it cover knee surgery?",
        "What is this document about?"
    ]
}

res = requests.post(url, headers=headers, json=payload)

print("Status Code:", res.status_code)
# Try to parse JSON safely
try:
    data = res.json()
    print("Parsed JSON:\n", data)
except Exception as e:
    print("❌ Error parsing JSON:", e)