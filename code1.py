# Hash Function and Hash Table Implementation in Python
# Defining hash table with size
# Initially the hash table is empty
hashTable = [None] * 10

# Defining hash function 'hashFunction'
def hashFunction(key):
    return key % len(hashTable) # Uses Division Method
# print(hashFunction(10))

# Inserting data into the hash table
# Defining the function 'insert'
def insert(hashTable, key, value):
    hashKey = hashFunction(key)
    hashTable[hashKey] = value

# Insert the data in the hash table
insert(hashTable, 10, 'Bangladesh')
insert(hashTable, 25, 'USA')

# Print the values in the hash table:
print('Output:\n')
print(hashTable)