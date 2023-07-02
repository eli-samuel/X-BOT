import discord
import os
import random
import asyncio
from datetime import datetime, timedelta

import codecs
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict
import codecs
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets, transforms
import torchvision
import torch
from torch.nn.modules.activation import Softmax
from torchsummary import summary
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

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
        global prev_day_of_week
        if (current_day_of_week - prev_day_of_week >=1) or (current_day_of_week == 0 and prev_day_of_week == 6): # if a day ("24 hours") has passed
            prev_day_of_week = current_day_of_week
            if (current_time.hour >= 14): # if time is past 9 am - time in UTC - 6 am EST is 11 AM UTC
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


def write_to_file(strings, path):
    """
    Writes a list of strings to a file at the specified path.

    Args:
        strings (list of str): The strings to write to the file.
        path (str): The path to the file to write.

    Returns:
        None
    """
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for s in strings:
            f.write(s + "\n")


def read_from_file(path):
    """
    Reads the contents of a file at the specified path into a list of strings.

    Args:
        path (str): The path to the file to read.

    Returns:
        list of str: The lines of the file, stripped of leading and trailing whitespace.
    """
    with codecs.open(path, 'r', encoding='utf-8') as f:
        strings = [line.strip() for line in f.readlines()]
    return strings


def add_masking_padding(data, pad_value=-1):
    """
    Adds padding to a batch of variable-length sequences and returns a mask indicating which values are real and which
    are padding.

    Args:
        data (list of list of int): The batch of variable-length sequences to pad.
        pad_value (int, optional): The value to use for padding. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - list of list of int: The padded batch of sequences.
            - list of list of int: The mask indicating which values are real and which are padding.
    """
    output = []
    # Get the length of the longest string in the batch
    max_len = max([len(string) for string in data])
    print(
        f"The longest sentence is {max_len} characters long. Adding padding to all shorter strings.")

    # Create a mask for the padding values
    mask = [pad_value] * len(data)
    for i in range(len(data)):
        mask[i] = [pad_value] * max_len

    # Pad each string with the padding value and set the corresponding mask values
    for i, string in enumerate(data):
        padded = [pad_value] * max_len
        for j in range(len(string)):
            padded[j] = string[j]
            mask[i][j] = 1
        output.append(padded)

    return output, mask