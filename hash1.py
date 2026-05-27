import hashlib
text = "Hello World"
data = text.encode('utf-8')
hash_value = hashlib.sha512(data).hexdigest()

print(hash_value)