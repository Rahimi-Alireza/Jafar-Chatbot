import discord
from colorama import Fore, Style
import main
PREFIX = "~"
CHANNEL = "AI"
class MyClient(discord.Client):
    def __init__(self, words, pat_y):
        self.words = words
        self.pat = pat_y
    async def on_ready(self):
        print('Logged on as', Fore.GREEN + self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return
        if message.channel != CHANNEL:
            return 
        if message.content.startswith(PREFIX):
            m = message.content
            answer = main.chat(self.words, self.pat_y,m)
            await message.channel.send('**BOT**:',answer)

    
def run(words, pat_y):
    client = MyClient()
    with open('token.secret', 'r') as file:
        token = file.read()
    client.run(token)