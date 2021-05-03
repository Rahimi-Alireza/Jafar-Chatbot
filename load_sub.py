import os
from colorama import Fore , Style

PATH = "Data/"  # dir name
DELETED_CHAR = '0123456789:-_.><?~!@"\';,؟\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

data = []

# for all files inside PATH
for i in os.listdir(PATH):
    file_path = PATH + i
    with open(file_path, "r", encoding="utf-8") as file:
        print('Opening file: ' , end='')
        print(Fore.GREEN + file_path)
        content = file.read()
        # Delete timestamp and other stuff
        for ch in DELETED_CHAR:
            content = content.replace(ch, "")
        lines = content.split("\n")
    print(Style.RESET_ALL , end='') #Reset the style of CLI
    for j in lines:
        if j.isspace() or j.startswith('\n') or j=='': #DO NOT Add whitespaces to our training data
            continue
        data.append(j)

print(Fore.BLUE + str(len(data)) + ' Sentences loaded') #Log how many files were loaded
print(Fore.RED + 'Ready to start learning')





