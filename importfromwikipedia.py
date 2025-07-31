import wikipedia
import re

# 1️⃣ Choose your topic
topic = "Artificial Intelligence"  # You can change this

# 2️⃣ Get the content
content = wikipedia.page(topic).content

# 3️⃣ Clean the text
# Remove brackets [1], [2] ...
cleaned_text = re.sub(r'\[[0-9]+\]', '', content)

# Remove multiple newlines
cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

# 4️⃣ Save to data.txt
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("✅ data.txt has been created successfully!")

