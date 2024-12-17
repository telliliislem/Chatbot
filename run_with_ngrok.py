from pyngrok import ngrok
import os

# Start ngrok tunnel
public_url = ngrok.connect(8502)
print(f"Application accessible à l'URL : {public_url}")

# Run the Streamlit app
os.system("streamlit run app.py --server.port 8502")
