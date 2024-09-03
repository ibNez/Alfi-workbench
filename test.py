import ollama
from ollama import Client
from dotenv import load_dotenv
import os
import redis

load_dotenv()

def clean_redis_url(s, char):
    parts = s.split(char, 2)
    return char.join(parts[:2])

TOKEN = os.getenv('APP_REDIS_DSN')
print(TOKEN)
redis_url = clean_redis_url(os.getenv('APP_REDIS_DSN'), ":")
print(redis_url)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

redis_client.set('foo', 'bar')
# True
print(redis_client.get('foo'))
# bar

print(redis_client.exists("blue"))

#client = Client(host='http://ollama:11434')
#llmmodel="llama3"
#client.pull(llmmodel)

#from langchain_ollama import OllamaLLM

#model = OllamaLLM(model="llama3",
 #                base_url="http://ollama:11434")
#response = model.invoke("Come up with 10 names for a song about parrots")
#print(response)
