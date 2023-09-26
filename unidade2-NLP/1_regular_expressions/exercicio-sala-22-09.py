import re

p1 = r'(\w+|\?|!)'
p2 = r'(\w+|#\d+|\?|!)'
p3 = r'(#\d\w+\?!)' # \ = 'e' - imediatamente após
p4 = r'\s+'

patterns = [p1, p2, p3, p4]

[print(re.findall(p, "SOLDIER #1: Found them? In Mercea? The coconut's tropical!")) for p in patterns]
