score_file = open("score.txt", "r", encoding="utf8")

lines = score_file.read()

print(lines)
# for line in lines:
#     print(line, end="")

score_file.close()