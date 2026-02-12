from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite"
)

response=model.invoke("What is the capital of France?")
print(response)