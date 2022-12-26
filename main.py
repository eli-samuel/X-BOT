import discord
# import os
import random

intents = discord.Intents.none()
intents.messages = True
intents.guild_messages = True
intents.reactions = True

# intents = discord.Intents.all()
# TOKEN = os.environ['TOKEN']
TOKEN = "MTA1NzA2NDUxMzY5NTkxMTk5Ng.GSGA-a.XF8ikLHRORdH2_n4Cm5XbL-kgHXU-S008IRwRk"
client = discord.Client(intents=intents, token=TOKEN)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    rand = random.uniform(0, 100)
    print(f"Generated {round(rand, 3)}")
    print(message)

    if message.author == client.user:
        return

    if message.author.id == 693691418308378644:
        print(f"{message.author} sent a message")
        await message.add_reaction("❌")
        await(message.add_reaction("👎"))

    elif message.author.id == 321458194071158784:
        print(f"{message.author} sent a message")
        await message.add_reaction("✅")

    elif message.author.id == 130456305126211585:
        if rand > 95:
            await message.channel.send("uwu eli you are so cute 👉👈 😳")

    elif "shut up" in message.content:
        await message.channel.send("yeah shut up")

    else:
        if rand < 2:
            print(f"{message.author} just got f randomed lmao")
            await message.add_reaction("🇫")
        elif rand < 4:
            print(f"{message.author} just got heart randomed lmao")
            await message.add_reaction("❤️")

client.run(TOKEN)

