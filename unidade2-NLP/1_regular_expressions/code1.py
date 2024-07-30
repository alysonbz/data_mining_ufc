import re
# Write a pattern to match sentence endings: sentence_endings
my_string = "Let's write RegEx!  Won't that be fun?  " \
            "I sure think so.  Can you find 4 sentences? " \
            "Or perhaps, all 19 words?"

sentence_endings = r"\!"

# Split my_string on sentence endings and print the result
print(re.split(my_string, sentence_endings))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.split(my_string, capitalized_words))

# Split my_string on spaces and print the result
spaces =  r"\s+"
print(re.split(my_string, spaces))

# Find all digits in my_string and print the result
digits = r"\d"
print(re.split(my_string, digits))
