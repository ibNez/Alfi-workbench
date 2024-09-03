import discord
from discord.ext import commands
import ollama
from ollama import Client
import os
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

load_dotenv()
ollama_url = os.getenv('APP_OLLAMA_URL')
ollama_client = Client(host=ollama_url)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

llmmodel="llama3"
list_of_models = ["llama2", "llama3", "phi3", "mistral", "codegemma", "gemma"]
TOKEN = os.getenv('DISCORD_TOKEN')
chat_messages = []
system_message="Response like you are pretending you are a human named Alfi. You may never mention you are pretending or that you are an AI. Format your response using Markdown text format."

subconscious = {}  # Active Messages In Memory
subconscious['username'] = chat_messages

memories = []
long_term_memory = {}

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
ollama_client.pull(llmmodel)

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
    global subconscious
    global ollama_url
    
    for user in subconscious:
        user_memories = {}
        chat_messages = []
        # Read memories from file if they exist for this user.
        if redis_client.exists(user) == 1:
            user_memories = redis_client.exists(user)
        else:
            print("No memories found for current user.")

        chat_messages = subconscious[user]
        if len(chat_messages) <= 1:
            continue

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
        redis_client.hset(user, mapping={
            'user_memories': user_memories,
        })
        subconscious[user] = []


    return

@bot.event
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



@bot.command(name='Alfi', help="Command used to talk to AI Alfi.")
@commands.has_role('ENFORCER')
async def on_message(ctx, paragraph, error):
    global subconscious

    chat_messages = []
    memory = {}
    user_memories = {}

    # Read memories from file if they exist for this user.
    try:
        with open(ctx.author.name, 'r') as file:
            user_memories = json.load(file)
    except:
        print("No memories found for current user.")

    # print(ctx.author.name)
    # print(ctx.message.content)

    # Check to see if we have spoken with this user before.
    if ctx.author.name in subconscious:
        if user_memories:
            print("These are the users memories: " + str(user_memories[ctx.author.name]))
            chat_messages.extend(user_memories[ctx.author.name])
        chat_messages.extend(subconscious[ctx.author.name])
    else:
        # Setup a default convesational rules for all users
        subconscious[ctx.author.name] = [create_message(system_message, 'system'),
                                       create_message('My name is '+ ctx.author.name, 'user'),
                                       create_message('Nice to meet you '+ ctx.author.name, 'assistant')]
        chat_messages = subconscious[ctx.author.name]
        if user_memories:
            chat_messages.extend(user_memories[ctx.author.name])

    # Form our conext for Alfi remove Alfi command from string.
    message_to_send = ctx.message.content.replace("!Alfi ", "") 
    chat_messages.append(create_message(message_to_send, 'user'))
    print("Current Change Messages: "+str(chat_messages))
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
        
        print(chunk['message'])

        if '\n' in chunk['message']['content']:
            print("Created Sentence: "+sentence)
            paragraph = paragraph + sentence + '\n'
            print("Paragraph: "+paragraph)
            if len(paragraph) >= 1500:
                print("Paragraph snippet being sent: "+paragraph)
                await ctx.send(paragraph)
                chat_messages.append(create_message(paragraph, 'assistant'))
                subconscious[ctx.author.name] = chat_messages
                paragraph = ''
            sentence = ''
        else:
            sentence = sentence + chunk['message']['content']
    paragraph = paragraph + sentence
    if len(paragraph) >= 1:
        print("Paragraph being sent: "+paragraph)
        await ctx.send(paragraph)
        chat_messages.append(create_message(paragraph, 'assistant'))
        subconscious[ctx.author.name] = chat_messages
    #print(str(subconscious))

    # Save Memories to longterm storage
    if len(subconscious) > 100:
        chat_messages.append(create_message('Can you summarize all our conversation keeping track of who said what?', 'user'))
        print("Current Chat Messages: "+str(chat_messages))
        stream = ollama_client.chat(
            model=llmmodel,
            messages = chat_messages,
            options = {
            #'temperature': 1.5, # very creative
            'temperature': 0 # very conservative (good for coding and correct syntax)
            },
            stream=True
        )

        for chunk in stream:    
            if '\n' in chunk['message']['content']:
                paragraph = paragraph + sentence + '\n'
                sentence = ''
            else:
                sentence = sentence + chunk['message']['content']
        paragraph = paragraph + sentence
        
        if user_memories:
            user_memories[ctx.author.name].append(create_message(paragraph, 'assistant'))
        else:
            user_memories[ctx.author.name] = [create_message(paragraph, 'assistant')]
        print("Saving Users Memories to file.")
        with open(ctx.author.name, 'w') as file:
            json.dump(user_memories, file)
        subconscious[ctx.author.name] = []

bot.run(TOKEN)