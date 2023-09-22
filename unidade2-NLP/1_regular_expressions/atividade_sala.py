import re
# A | é para determinar um ponto de parada
pattern1 = '(\w+|\?|)'
pattern2 = '(\w+|#\d+|\?|!)'
pattern3 = '(#\d\w+\?!)'
pattern4 = '\s+'
print(re.findall(pattern2, "SOLDIER #1: Found them? In Mercea? The coconut's tropical"))