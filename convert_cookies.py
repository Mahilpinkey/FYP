import json
import os

# Input and output file paths
input_file = r"C:\Users\priya\Downloads\www.youtube.com_22-08-2025.json"
output_file = os.path.join(os.getcwd(), "cookies.txt")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract the list of cookies
cookies = data.get("cookies", data)  # handles both wrapped and direct list cases

with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Netscape HTTP Cookie File\n")
    for cookie in cookies:
        domain = cookie.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = cookie.get("path", "/")
        secure = "TRUE" if cookie.get("secure", False) else "FALSE"
        expiry = str(int(cookie.get("expirationDate", 0))) if "expirationDate" in cookie else "0"
        name = cookie.get("name", "")
        value = cookie.get("value", "")

        line = f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n"
        f.write(line)

print(f"Cookies saved to {output_file}")
