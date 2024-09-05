import ollama
from ollama import Client
from dotenv import load_dotenv
import os
import redis
import json

load_dotenv()

def clean_redis_url(s, char):
    parts = s.split(char, 2)
    return char.join(parts[:2])

TOKEN = os.getenv('APP_REDIS_DSN')
print(TOKEN)
redis_url = clean_redis_url(os.getenv('APP_REDIS_DSN'), ":")
print(redis_url)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)


user_memories = {'_.nez._': [{'role': 'assistant', 
                              'content': 'That... to add anything else to our conversation!'
                              }
                              ]}
chat_messages = []
user = list(user_memories.keys())[0]

json_string = json.dumps(user_memories)

redis_client.hset(user, mapping={
    'json': json_string,
})

retrieved_value=redis_client.hgetall(user)

old_memories = json.loads(retrieved_value['json'])

print(str(old_memories))

#client = Client(host='http://ollama:11434')
#llmmodel="llama3"
#client.pull(llmmodel)

#from langchain_ollama import OllamaLLM

#model = OllamaLLM(model="llama3",
 #                base_url="http://ollama:11434")
#response = model.invoke("Come up with 10 names for a song about parrots")
#print(response)
