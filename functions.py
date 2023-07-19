import discord
import os
import random
import asyncio
from datetime import datetime, timedelta

global prev_day_of_week
prev_day_of_week = 0

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

async def send_gm(message):
        current_time = datetime.now()
        current_day_of_week = datetime.now().weekday()
        print("The current day is: ", current_day_of_week)
        global prev_day_of_week
        print("Previous day of week is: ", prev_day_of_week)
        print("Current time is: ", current_time.hour)
        if (current_day_of_week - prev_day_of_week >=1) or (current_day_of_week == 0 and prev_day_of_week == 6): # if a day ("24 hours") has passed
            prev_day_of_week = current_day_of_week
            print("SENDING MESSAGE, PREV DAY IS NOW ", prev_day_of_week )
            if (current_time.hour >= 9): # if time is past 9 am - time in UTC - 6 am EST is 11 AM UTC
                if (current_day_of_week == 0):# monday
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/blessings-god-bless-family-sparkle-gif-24714833")
                elif (current_day_of_week == 1):
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/tuesday-happy-blessings-good-morning-gif-23349785")
                elif (current_day_of_week == 2):
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/happy-wednesday-bahonon-jayjay-opely-greetings-gif-12105891")
                elif (current_day_of_week == 3):
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/141thur-gif-25653641")
                elif (current_day_of_week == 4):
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/friday-happy-love-gif-25332949")
                elif (current_day_of_week == 5):
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/happy-day-gif-25988861")
                elif (current_day_of_week == 6): #sunday
                    bot_message = await message.channel.send(f"Good morning! \n https://tenor.com/view/happy-sunday-gif-26115569")
                else: #impossible?
                    bot_message = await message.channel.send(f" Huh ? \n https://tenor.com/view/cat-sniff-gif-26264120")

async def join_voice(message, audio_path):
    connected = message.author.voice.channel
    if not connected:
        await message.channel.send("You aren't in a voice channel wtf")
        return
    
    print(f"Connected to: {connected}")
    
    audio_source = discord.FFmpegPCMAudio(audio_path)
    print(f"Audio source: {audio_source}")

    print("A")

    try:
        voice = await connected.connect()
    except asyncio.TimeoutError:
        await message.channel.send("Failed to connect to voice channel")
        return
    
    print("B")
    voice.play(audio_source)
    print("C")

    # wait for the audio to finish playing
    while voice.is_playing():
        print("D")
        await asyncio.sleep(1)

    await voice.disconnect(force=True)

async def leave_voice(message):
    voice_client = message.guild.voice_client
    if not voice_client:
        await message.channel.send("I'm not currently in a voice channel.")
        return

    for channel in message.guild.voice_channels:
        if voice_client.channel == channel:
            await voice_client.disconnect(force=True)
            await message.channel.send(f"Leaving voice channel: {channel.name}")
            return

    await message.channel.send("I'm not currently in a voice channel.")
