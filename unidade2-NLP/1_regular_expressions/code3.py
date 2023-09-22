from nltk.tokenize import sent_tokenize
from src.utils import get_sample_Santo_Graal
import re
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search(r"coconuts", scene_one)

# Print the start and end indexes of match
if match:
    print("inicial:", match.start())
    print("final:", match.end())
else:
    print("Palavra 'coconuts' não encontrada.")

print(match)

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*?\]"

# Use re.search to find the first text in square brackets
match = re.search(pattern1, scene_one)
if match:
    print("Texto entre colchetes:", match.group())
else:
    print("Nenhum texto encontrado.")
# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"^\w+:"
fourth_sentence = sentences[3]
match = re.match(pattern2, fourth_sentence)
if match:
    print("Notação na quarta sentença:", match.group())
else:
    print("Notação não encontrada na quarta sentença.")