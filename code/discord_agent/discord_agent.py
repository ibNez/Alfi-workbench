# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import discord
from discord.ext import commands

import ollama
from ollama import Client

import os
import copy
from dotenv import load_dotenv
import json

import redis

from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from optimum.intel import OVStableDiffusionXLPipeline
from diffusers.utils import export_to_video

import torch
import gc

from PIL import Image
import requests

import pyttsx3

load_dotenv()
GUILD_ID = 941395439708667915
CHANNEL_ID = 997679589025390622

ollama_url = os.getenv('APP_OLLAMA_URL')
ollama_client = Client(host=ollama_url)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

llmmodel="llama3"
list_of_models = ["llama2", "llama3", "phi3", "mistral", "codegemma", "gemma"]
TOKEN = os.getenv('DISCORD_TOKEN')
chat_messages = []
system_message="Your name is Alfi. You are a helpful AI assistant. You do your best to be honest and use creditable information. If you do not know something you say you do not know."

conscious_thoughts = {}  # Active Messages In Memory
conscious_thoughts['username'] = chat_messages

memories = []
long_term_memory = {}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
engine = pyttsx3.init()
ollama_client.pull(llmmodel)

# Create audio output file.
def text_to_speech(text, filename):
    engine.setProperty('voice', 'slt')  # Use the Salli voice
    engine.setProperty('rate', 150)
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Defining a function to create new messages
def create_message(message, role):
  return {
    'role': role,
    'content': message
  }
# Set Default State of Alfi
chat_messages.append(create_message(system_message, 'system'))

#Start up Discord Bot Server listener
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    await bot.change_presence(activity=discord.Game(name="with your Mind!"))

    # Custom server connected message
    for guild in bot.guilds:
        print(f'Connected to server: {guild.name} (ID: {guild.id})')

# This method allow us to ask Alfi to download and use another LLM
@bot.command(name='AlfiUpdate', help='Used to update Alfi child LLM model')
@commands.has_role('ENFORCER')
async def update_model(ctx, error):
    global ollama_client

    print(ctx.message.content.replace("!AlfiUpdate ", ""))
    command = ctx.message.content.replace("!AlfiUpdate ", "")
    if command in list_of_models:
        try:
            llmmodel = command
            ollama_client.pull(llmmodel)
            await ctx.send("Model updated to: " + command)
            print("Model updated to: " + command)
        except Exception as e:
            print("Error: " + str(e))
            error = str(e)
            await ctx.send(error)
    else:
        print("Model not applied, please use one of the following: " + ', '.join(list_of_models))
        await ctx.send("Model not applied, please use one of the following: " + ', '.join(list_of_models))

#TODO Create a fuction for storing older messages to long term memory.  This method should call Alfi to summerize the current set of conversations and then save a new summary to the users
# perminate memory.  This will be saved to a file using pickle

#TODO Create a function that can be called by discord command to force all subconsion memories to long team storage.
# Use this message to unload current conversation into history > redis db
@bot.command(name='Sleep', help="Prepare Alfi for sleeping and store memories to long term.")
@commands.has_role('ENFORCER')
async def sleep(ctx, paragraph, error):
    global conscious_thoughts
    global ollama_url

    user = ctx.author.name  
    user_memories = {}
    chat_messages = []
    
    # Read memories from file if they exist for this user.
    if redis_client.exists(user) == 1:
        retrieved_value=redis_client.hgetall(user)

        user_memories = json.loads(retrieved_value['json'])
    else:
        print("No memories found for current user.")

    chat_messages = conscious_thoughts[user]
    if len(chat_messages) <= 1:
        return

    if user_memories:
        chat_messages = [i for i in chat_messages if i not in user_memories[user]]

    #TODO May need to remove old memorys compare to current memories and remove from list first.?

    chat_messages.append(create_message('Can you summarize all our conversation keeping track of who said what?', 'user'))
    print("Asking Alfi for a Summary of current memory")
    stream = ollama_client.chat(
        model=llmmodel,
        messages = chat_messages,
        options = {
        #'temperature': 1.5, # very creative
        'temperature': 0 # very conservative (good for coding and correct syntax)
        },
        stream=True
    )
    sentence = ''
    paragraph = ''
    for chunk in stream:    
        if '\n' in chunk['message']['content']:
            paragraph = paragraph + sentence + '\n'
            sentence = ''
        else:
            sentence = sentence + chunk['message']['content']
    paragraph = paragraph + sentence
    
    if user_memories:
        user_memories[user].append(create_message(paragraph, 'assistant'))
    else:
        user_memories[user] = [create_message(paragraph, 'assistant')]
    print("Saving Users Memories to Redis.")
    print("Chat Messages: " + str(chat_messages))
    print("UserID: " + user)
    json_string = json.dumps(user_memories)

    redis_client.hset(user, mapping={
        'json': json_string,
    })

    #with open(ctx.author.name, 'w') as file:
    #    json.dump(user_memories, file)
    #subconscious[ctx.author.name] = []
    #with open(user, 'w') as file:
    #    json.dump(user_memories, file)
    conscious_thoughts[user] = []


@bot.event
@commands.has_role('ENFORCER')
async def on_reaction_add(reaction, user):
    # Check if the reaction is the desired emoji 
    if str(reaction.emoji) == "🖼️":
        print(str(torch.cuda.memory_summary(device=None, abbreviated=False)))
        torch.cuda.empty_cache()
        # Get the original message
        original_message = reaction.message

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # Create a pipeline for diffution
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")

        pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)

        # Form a prompt from the previous message
        prompt = str(original_message.content).replace("!Alfi ", "")

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 100
        high_noise_frac = 0.8
        batch_size = 1

        # run both experts
        try:
            image = pipeline(prompt).images[0]
            # image = pipe(prompt=prompt, num_inference_steps=n_steps).images[0]
        except Exception as e:
            print("Error fetching image: " + str(e))
            torch.cuda.empty_cache()
            gc.collect()
            return

        #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        image.save("diffusion_images/generated.png")
        await original_message.channel.send("Image Created!  🖼️", file=discord.File("diffusion_images/generated.png"))
        torch.cuda.empty_cache()
        gc.collect()

    # Check if the reaction is the desired emoji 
    if str(reaction.emoji) == "🎥":
        torch.cuda.empty_cache()
        image_url = ""
        if len(reaction.message.attachments) > 0:
            attachment = reaction.message.attachments[0]
        else:
            return
        if attachment.filename.endswith(".jpg") or attachment.filename.endswith(".jpeg") or attachment.filename.endswith(".png") or attachment.filename.endswith(".webp") or attachment.filename.endswith(".gif"):
            image_url = attachment.url
        elif "https://images-ext-1.discordapp.net" in reaction.message.content or "https://tenor.com/view/" in reaction.message.content:
            image_url = reaction.message.content
        # Create a pipeline for diffution
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16" if torch.cuda.is_available() else "fp32"
        )
        #pipe.to("cuda")
        #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()


        # Load the conditioning image
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = image.resize((256, 256))

        generator = torch.manual_seed(42)
        
        frames = pipe(image, decode_chunk_size=1, generator=generator, num_frames=25).frames[0]

        export_to_video(frames, "diffusion_movies/generated.mp4", fps=7)
        await original_message.channel.send("Movie Created!  🎥", file=discord.File("diffusion_images/generated.mp4"))
        torch.cuda.empty_cache()
        gc.collect()

#@bot.event
@bot.command(name='MakeVideo', help="Prompt Alfi to draw a picture.")
@commands.has_role('ENFORCER')
async def on_message(self, message):
    if len(message.attachments) > 0:
        attachment = message.attachments[0]
    else:
        return
    if attachment.filename.endswith(".jpg") or attachment.filename.endswith(".jpeg") or attachment.filename.endswith(".png") or attachment.filename.endswith(".webp") or attachment.filename.endswith(".gif"):
        self.image = attachment.url
    elif "https://images-ext-1.discordapp.net" in message.content or "https://tenor.com/view/" in message.content:
        self.image = message.content

    pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Load the conditioning image
    image = self.image.resize((1024, 576))

    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

    export_to_video(frames, "diffusion_movies/generated.mp4", fps=7)

    e = discord.Embed()
    e.set_image(url=image)
    await ctx.send(embed=e)

@bot.command(name='AlfiStop', help="Interupt Alfi talking.")
@commands.has_role('ENFORCER')
async def alfistop(ctx, error):
    if ctx.voice_client:
        ctx.voice_client.stop()
        await ctx.voice_client.disconnect()
        #engine.stop()
        await ctx.send("Stopped speaking and disconnected from the voice channel.")
    else:
        await ctx.send("I'm not connected to a voice channel.")

@bot.command(name='Alfi', help="Command used to talk to AI Alfi.")
@commands.has_role('ENFORCER')
async def on_message(ctx, paragraph, error):
    # conscious_thoughts type = dict = Current log of user interactions stored in RAM
    # structure: 
    #   
    # short_term_memory type = dict = History of user interactions Stored in Redis DB
    # structure:
    #   {'json': json.dump(chat_messages)}
    global conscious_thoughts
    chat_messages = []
    short_term_memory = {}
    user = ctx.author.name
    filename = user+".wav"

    # Read memories from file if they exist for this user. 
    if redis_client.exists(user) == 1:
        
        retrieved_value=redis_client.hgetall(user)

        short_term_memory = json.loads(retrieved_value['json'])
        #with open(ctx.author.name, 'r') as file:
        #    user_memories = json.load(file)
    else:
        print("No memories found for current user.")

    # print(ctx.author.name)
    # print(ctx.message.content)

    # Merge conscious_thoughts and short_term_memory into chat_messages.
    # Check to see if we have conversations stored in RAM(conscious_thoughts) conversations.
    if ctx.author.name in conscious_thoughts:
        print("User, " + ctx.author.name + ", has conscous thoughts stored in RAM.")
        # Check to see if we have conversations stored in Redis(user_shorterm_memory). If there are we want them 
        # to be first in the chat_messages variable.
        if short_term_memory:
            print("User, " + ctx.author.name + ", has shorterm memories stored in redis. Adding them to chat_messaages.")
            chat_messages.extend(short_term_memory[ctx.author.name])
            short_term_memory = {}
        # Load up chat messages to be handed to the LLM
        print("Adding conscious_thoughts to the chat messages after any prevous found shorterm memories.")
        chat_messages.extend(conscious_thoughts[ctx.author.name])
    else:
        # If there are no subconcious or memories we need to set a default. If there are memories we need to use them as they contain the default?  But for how long?
        print("User, " + ctx.author.name + ", has no current conscous thoughts stored in RAM.")
        if short_term_memory:
            print("User, " + ctx.author.name + ", has shorterm memories stored in redis. Adding them to chat_messaages.")
            chat_messages.extend(short_term_memory[ctx.author.name])
            short_term_memory = {}
            conscious_thoughts[ctx.author.name] = []
        else:
            # Setup a default convesational rules for all users and add to chat messages to be handed to the LLM
            print("User, " + ctx.author.name + ", has never spoken with Alfi before. Creating default conversation parameters and creating chat_messages.")
            conscious_thoughts[ctx.author.name] = [create_message(system_message, 'system'),
                                        create_message('My name is '+ ctx.author.name, 'user'),
                                        create_message('Nice to meet you '+ ctx.author.name + ". My Name is Alfi.", 'assistant')]
            chat_messages = copy.deepcopy(conscious_thoughts[ctx.author.name])
        
    # Format our context for Alfi and remove !Alfi command from string.
    message_to_send = ctx.message.content.replace("!Alfi ", "")
    print("User, " + ctx.author.name + ", adding new user text to chat_messages./n Submitting to LLM...") 
    chat_messages.append(create_message(message_to_send, 'user'))
    conscious_thoughts[ctx.author.name].append(create_message(message_to_send, 'user'))
    stream = ollama_client.chat(
        model=llmmodel,
        messages = chat_messages,
        options = {
        #'temperature': 1.5, # very creative
        'temperature': 0 # very conservative (good for coding and correct syntax)
        },
        stream=True
    )

    sentence = ''
    paragraph = ''
    llm_response = ''

    print("Reading response from stream.")
    for chunk in stream:
        llm_response = llm_response + chunk['message']['content']
        print("Message in Chunk of stream: " + str(chunk['message']))
        if '\n' in chunk['message']['content']:
            print("New Line Symbol found in content of message chunk: " + chunk['message']['content'])
            if sentence != '':
                # Adding new sentence to Paragraph
                paragraph = paragraph + sentence + chunk['message']['content'].replace("\n", "") + '\n'
                sentence = ''
            # Check if paragraph is too big and need to be sent in parts.
            if len(paragraph) >= 1500:
                print("Paragraph greater than 1500 character.\n Sending snippet of response: " + paragraph)
                await ctx.send(paragraph)
                paragraph = ''
                sentence = ''
        else:
            # No new line found. adding chunk to sentence.
            sentence = sentence + chunk['message']['content']
    # Add the remaining sentence to the paragraph
    paragraph = paragraph + sentence
    del sentence

    # Check our Paragraph response has unsent value and send it back to Discord.
    if len(paragraph) >= 1:
        print("Paragraph being sent: " + paragraph)
        await ctx.send(paragraph)
        
        del paragraph
        del chat_messages
        # Append new response to concious thoughts
    text_to_speech(llm_response, filename)
    conscious_thoughts[ctx.author.name].append(create_message(llm_response, 'assistant'))
    if ctx.author.voice:
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            voice_client = await channel.connect()
        else:
            await ctx.send("You are not connected to a voice channel.")
            return
        source = discord.FFmpegPCMAudio(filename)
        voice_client.play(source)
        while voice_client.is_playing():
            await discord.utils.sleep(1)
        await voice_client.disconnect()
    else:
        await ctx.send("You are not connected to a voice channel.")
    #print(str(subconscious))

    # Save Memories to short_term_memory storage in RDS
    if len(conscious_thoughts[ctx.author.name]) > 10:
        
        if short_term_memory:
            # Persist Conversation to Memory stored in Redis
            short_term_memory[ctx.author.name].append(conscious_thoughts[ctx.author.name])
        else:
            # Expand the Conversation Memory stored in Redis
            short_term_memory[ctx.author.name] = conscious_thoughts[ctx.author.name]
        
        print("Saving Users Memories to redis.")

        json_string = json.dumps(short_term_memory)

        redis_client.hset(user, mapping={
            'json': json_string,
        })

        #with open(ctx.author.name, 'w') as file:
        #    json.dump(user_memories, file)
        conscious_thoughts.pop(ctx.author.name, None)

bot.run(TOKEN)