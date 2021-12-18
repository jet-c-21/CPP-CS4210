import re


x = '3.7'

if re.match(r'^-?\d+(?:\.\d+)$', x) is not None:
    print("it's float")

x = '9'
if re.match(r'^[-+]?[0-9]+$', x) is not None:
    print("it's int")