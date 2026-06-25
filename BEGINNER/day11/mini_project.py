# Day 11 Mini-Project — Library Book System (OOP I)

class Book:
    def __init__(self, title, author, isbn, copies=1):
        self.title  = title
        self.author = author
        self.isbn   = isbn
        self.copies = copies
        self._borrowed = 0

    def borrow(self):
        if self._borrowed >= self.copies:
            raise ValueError(f"No copies of '{self.title}' available")
        self._borrowed += 1

    def return_book(self):
        if self._borrowed == 0:
            raise ValueError("All copies already returned")
        self._borrowed -= 1

    @property
    def available(self): return self.copies - self._borrowed

    def __str__(self):
        return f"[{self.isbn}] '{self.title}' by {self.author} (avail: {self.available}/{self.copies})"


class Library:
    def __init__(self, name):
        self.name  = name
        self._books = {}   # isbn → Book

    def add_book(self, book):
        self._books[book.isbn] = book
        print(f"Added: {book}")

    def find(self, query):
        q = query.lower()
        return [b for b in self._books.values()
                if q in b.title.lower() or q in b.author.lower()]

    def borrow(self, isbn):
        book = self._books.get(isbn)
        if not book: raise KeyError(f"ISBN {isbn} not found")
        book.borrow()
        print(f"Borrowed: {book.title}")

    def return_book(self, isbn):
        book = self._books.get(isbn)
        if not book: raise KeyError(f"ISBN {isbn} not found")
        book.return_book()
        print(f"Returned: {book.title}")

    def catalog(self):
        print(f"\n=== {self.name} Catalog ===")
        for book in self._books.values():
            print(" ", book)


lib = Library("City Library")
lib.add_book(Book("Python Crash Course", "Matthes", "001", 2))
lib.add_book(Book("Clean Code",          "Martin",  "002", 1))
lib.add_book(Book("The Pragmatic Programmer", "Hunt", "003", 1))

lib.catalog()
lib.borrow("001")
lib.borrow("001")

try:
    lib.borrow("001")
except ValueError as e:
    print(f"Error: {e}")

lib.return_book("001")
lib.catalog()

results = lib.find("python")
print("\nSearch 'python':", [b.title for b in results])
