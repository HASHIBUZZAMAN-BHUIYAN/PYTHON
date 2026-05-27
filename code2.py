# An example of Collision Occurs
# Collision occurs when multiple items get the same index
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
insert(hashTable, 25, 'Korea') # Collision occurs here; overwrites 'USA'

# Print the values in the hash table:
print('Output:\n')
print('The USA is replaced by Korea, because of the result of hash function for same keys.')
print(hashTable)