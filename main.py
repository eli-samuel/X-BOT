import discord
import os
import random
import asyncio
from datetime import datetime, timedelta
from functions import rand_spam_msg, remove_role, send_gm

from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("TOKEN")

intents = discord.Intents.all()

client = discord.Client(intents=intents, token=TOKEN)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message): # Runs actions the moment a message is sent in the server? 
    if message.author == client.user: # ignore messages sent by the bot itself  
        return

    await send_gm(message)

    #RNG for specific features, generates a number between 0-100
    rand = random.uniform(0, 100)
    print(f"{message.author} generated {round(rand, 3)} for message \"{message.content}\"")

    # Stuff for specific people
    if message.author.id == 321458194071158784: # Jocelyn's ID
        print(f"{message.author} sent a message")
        await message.add_reaction("❌")
        await(message.add_reaction("👎"))
    elif message.author.id == 693691418308378644: # Logang's ID 
        print(f"{message.author} sent a message")
        await message.add_reaction("✅")
    elif message.author.id == 130456305126211585: # Eli's ID 
        if rand == 97:
            await message.channel.send("uwu eli you are so cute 👉👈 😳")
        elif rand == 98: 
            await message.channel.send("eli please I want to bear your children")
        elif rand == 99:
            await message.channel.send("eli marry me I beg")

    # Stuff for general messages
    if "shut" in message.content.lower() and "up" in message.content.lower() or "stfu" in message.content.lower(): #check for any message containing "shut up"
        await message.channel.send("yeah shut up")
    else:  # for other messages not containing "shut up"
        if rand < 1: #more RNG 
            print(f"{message.author} just got f randomed lmao")
            await message.add_reaction("🇫")
        elif rand < 2: #more RNG 
            print(f"{message.author} just got heart randomed lmao")
            await message.add_reaction("❤️")
        elif rand < 3: #more RNG 
            print(f"{message.author} just got x randomed lmao")
            await message.add_reaction("❌")

    # period
    if "." == message.content.lower():
        print("PERIOD LOL")
        await message.channel.send("period")

    # 8 ball stuff
    if "xbot" in message.content.lower():
        print(f"8ball time for message: {message.content}")
        answers = random.randint(1,10)
        if message.content[-1] == '?':
            r = ""
            if answers == 1:
                r = "certainly"
            elif answers == 2:
                r = "damn right"
            elif answers == 3:
                r = "you may rely on it"
            elif answers == 4:
                r = "ask again later"
            elif answers == 5:
                r = "focus and ask a better question"
            elif answers == 6:
                r = "reply hazy, i forgot my glasses"
            elif answers == 7:
                r = "my reply is no"
            elif answers == 8:
                r = "hard no"
            elif answers == 9:
                r = "go away queer"
            elif answers == 10:
                r = "period mimi"
            await message.channel.send(r)
        else:
            if answers < 5:
                print("what?")
            else:
                print("is that a question?")
    
    # High stuff
    if message.content == "!hi": # wtf 
        role = discord.utils.get(message.guild.roles, id=1060309166767476887)
        print(f"Adding {role} to {message.author}")

        await message.author.add_roles(role)
        await message.add_reaction(":highaf:1000110789228777533>")
        bot_message = await message.channel.send(f"{message.author.mention}, you have 60 seconds to enter: https://discord.gg/jaTCB25aM6")

        await remove_role(message.author, role)

        await bot_message.delete()
        await message.delete()

    # Good morning at 3 pm response
    if message.content.lower() == "good morning":
        await message.channel.send("hi")
    
    # Good morning memes
    if message.content == "!settime":
        global prev_day_of_week
        prev_day_of_week  = datetime.now().weekday()
            
    # Check for spam
    messages = [] # message buffer 

    # Get the messages
    async for message in message.channel.history(limit=5):
        messages.append(message)

    # Check if the messages were sent by the same person in 10 seconds
    if (messages[0].author == messages[1].author == messages[2].author == messages[3].author == messages[4].author and (messages[0].created_at - messages[4].created_at).total_seconds() <= 10):
        await rand_spam_msg(message, (messages[0].created_at-messages[4].created_at).total_seconds())
        
client.run(TOKEN)
