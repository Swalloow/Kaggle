def get_web(url) :
    """input: URL, output: content"""
    import urllib.request
    response = urllib.request.urlopen(url)
    data = response.read()
    decoded = data.decode('utf-8')
    return decoded

url = input('webpage address : ')
content = get_web(url)
print(content)