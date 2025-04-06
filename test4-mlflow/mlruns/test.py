import chardet

with open('MLproject', 'rb') as f:
    result = chardet.detect(f.read(10000))
    print(result)