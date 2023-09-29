import re

pattern1 = '(\w+|\?|!)'
pattern2 = '(\w+|#\d+|\?|!)'
pattern3 = '(#\d\w+\?!)'
pattern4 = '\s+'

text = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"

print(re.findall(pattern1,  text))
print(re.findall(pattern2,  text))
print(re.findall(pattern3,  text))
print(re.findall(pattern4,  text))

