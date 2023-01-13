import discord
import os
import random
import asyncio

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

    #RNG for specific features, generates a number between 0-100
    rand = random.uniform(0, 100)
    print(f"{message.author} generated {round(rand, 3)} for message \"{message.content}\"")

    # Stuff for specific people
    if message.author.id == 693691418308378644: # Jocelyn's ID
        print(f"{message.author} sent a message")
        await message.add_reaction("❌")
        await(message.add_reaction("👎"))
        
    elif message.author.id == 321458194071158784: # Logang's ID 
        print(f"{message.author} sent a message")
        await message.add_reaction("✅")
        
    elif message.author.id == 130456305126211585: # Eli's ID 
        if rand > 97:
            if rand == 98:
                await message.channel.send("uwu eli you are so cute 👉👈 😳")
            elif rand == 99: 
                await message.channel.send("eli please I want to bear your children")
            else:
                await message.channel.send("eli marry me I beg")
    # Stuff for general messages
    if "shut" in message.content.lower() and "up" in message.content.lower() or "stfu" in message.content.lower(): #check for any message containing "shut up"
        await message.channel.send("yeah shut up")
    else:  # for other messages not containing "shut up"
        if rand < 2: #more RNG 
            print(f"{message.author} just got f randomed lmao")
            await message.add_reaction("🇫")
        elif rand < 4: #more RNG 
            print(f"{message.author} just got heart randomed lmao")
            await message.add_reaction("❤️")
        elif rand < 6: #more RNG 
            print(f"{message.author} just got x randomed lmao")
            await message.add_reaction("❌")
    
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

    # Check for spam
    messages = [] # message buffer 

    # Get the messages
    async for message in message.channel.history(limit=5):
        messages.append(message)

    # Check if the messages were sent by the same person in 10 seconds
    if (messages[0].author == messages[1].author == messages[2].author == messages[3].author == messages[4].author and (messages[0].created_at - messages[4].created_at).total_seconds() <= 10):
        await rand_spam_msg(message, (messages[0].created_at-messages[4].created_at).total_seconds())

async def rand_spam_msg(message, time):
    rand = round(random.uniform(0, 7), 1)

    print(f"Five messages were sent by the {message.author} in {time} seconds generating {rand}")

    if rand <= 2.:
        await message.channel.send(f"{message.author.mention}, please uninstall your send button")
    elif rand <= 4.:
        await message.channel.send(f"wtf {message.author.mention} stop spamming!")
    elif rand <= 5.:
        await message.channel.send(f"fyi {message.author.mention} just sent five messages in {time} seconds")
    elif rand <= 6.:
        await message.channel.send(f"https://tenor.com/view/spam-spam-intensifies-yum-gif-13948300")
    elif rand <= 6.5:
        await message.channel.send(f"{message.author.mention}")
        await message.channel.send(f"you")
        await message.channel.send(f"don't")
        await message.channel.send(f"need")
        await message.channel.send(f"to")
        await message.channel.send(f"type")
        await message.channel.send(f"like")
        await message.channel.send(f"this")
    elif rand <= 7:
        await message.channel.send(f"it took {message.author.mention} {time} seconds to send 5 messages, I last longer in bed and I'm not even real")
    

async def remove_role(member, role):
    await asyncio.sleep(60)
    print("Removing role from", member)
    await member.remove_roles(role)

client.run(TOKEN)
