import re
'''resp = re.match('mateus','mateus eh lindo')
print(resp)
resp1 = re.match('feio','mateus eh feio')
print(resp1)
word_regex = '\s+'
resp = re.split(word_regex,'semana de aula')
print(resp)
word_regex = r'[a-z]\w+'
resp = re.split(word_regex,'4Semana Quente! De Aula')
print(resp)'''
import re
pattern1 = '(\w+|\?|!)'
pattern2 = '(\w+|#\d+|\?\!)'
pattern3 = '(#\d\w+\?!)'
pattern4 = '\s+'
print(re.findall(pattern1,"SOLDIER #1: Found them? In mercea? The coconut's tropical"))
from nltk.tokenize import sent_tokenize
