import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()  # picks up GEMINI_API_KEY from env

# Upload the video file
video_file = client.files.upload(
    file="/home/avvk/Graxon/Graxon/parser/test_data/IMG_3509.MOV"
)

# Wait for processing (required for video)
while video_file.state == "PROCESSING":
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

if video_file.state == "FAILED":
    raise ValueError("Video processing failed")

prompt = """
Analyze this video and return:
1. A summary of what's happening
2. Chapter/scene breakdown with approximate timestamps
3. Any on-screen text
4. Key topics discussed
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[video_file, prompt],
)

print(response.text)

# Clean up
client.files.delete(name=video_file.name)
