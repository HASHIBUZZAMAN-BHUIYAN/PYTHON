# An example of collision resolution
# There are many solutions: 1. Linear Probing and 2. Chaining
# We are using here Chaining Method
# Defining hash table with size:
# Initially the hash table is empty and a nested list
hashTable = [[] for _ in range(10)]
# print(hashTable)

# Defining hash function:
def hashFunction(key):
    return key % len(hashTable) # Using Division Method
# print(hashFunction(10))

# Inserting data into the hash table:
# Defining the function 'insert'
def insert(hashTable, key, value):
    hashKey = hashFunction(key)
    hashTable[hashKey].append(value)

# Insert the data in the hash table
insert(hashTable, 10, 'Bangladesh')
insert(hashTable, 25, 'USA')
insert(hashTable, 25, 'Korea') # Chaining appends this to the existing inner list

# Print the values in the hash table:
print('Output:\n')
print('The USA and Korea are in the same list.')
print(hashTable)