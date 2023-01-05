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
async def on_message(message):
    rand = random.uniform(0, 100)
    print(f"Generated {round(rand, 3)}")
    print(message.author, "-", message.content)

    if message.author == client.user:
        return

    # Stuff for specific people
    if message.author.id == 693691418308378644:
        print(f"{message.author} sent a message")
        await message.add_reaction("❌")
        await(message.add_reaction("👎"))
    elif message.author.id == 321458194071158784:
        print(f"{message.author} sent a message")
        await message.add_reaction("✅")
    elif message.author.id == 130456305126211585:
        if rand > 97:
            await message.channel.send("uwu eli you are so cute 👉👈 😳")

    # Stuff for general messages
    if "shut" in message.content.lower() and "up" in message.content.lower() or "stfu" in message.content.lower():
        await message.channel.send("yeah shut up")
    else:
        if rand < 2:
            print(f"{message.author} just got f randomed lmao")
            await message.add_reaction("🇫")
        elif rand < 4:
            print(f"{message.author} just got heart randomed lmao")
            await message.add_reaction("❤️")
    
    # High stuff
    if message.content == "!hi":
        print("Someone is high")
        role = discord.utils.get(message.guild.roles, id=1060309166767476887)
        print(f"Adding {role} to {message.author}")
        await message.author.add_roles(role)
        await message.add_reaction(":highaf:1000110789228777533>")
        await message.delete()
        await remove_role(message.author, role)

async def remove_role(member, role):
    await asyncio.sleep(10)
    print("Removing role from", member)
    await member.remove_roles(role)

client.run(TOKEN)

