from collections import Counter

num_file = 9
mylist = []

for i in range(num_file):
    mylist.append([])
    with open('0' + str(i) + 'predictions_WR2021.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            mylist[i].append(ann)
    f.close()

for i in range(num_file):
    print(mylist[i])

count = 0
for i in range(len(mylist[0])):
    for j in range(num_file - 1):
        if mylist[j][i] != mylist[j + 1][i]:
            count += 1
            break

print('differences:{}'.format(count / len(mylist[0]) * 100))
count = 0
result = []
dismiss = []
for i in range(len(mylist[0])):
    temp = []
    result.append(mylist[0][i])
    for j in range(num_file):
        temp.append(mylist[j][i][-1])
    get = Counter(temp)
    if max(list(get.values())) <= num_file/1.5:
        print('\n')
        print(result[i])
        dismiss.append(result[i])
        print(get)
        t_list = list(get.values())
        val_key = max(t_list)
        target = list(get.keys())[list(get.values()).index(val_key)]
        str = list(result[i])
        str[-1] = target
        result[i] = ''.join(str)
        print(result[i])

f.close()
print(mylist[0])
print('after voting:')
print(result)
# print(dismiss)
count = 0.0
for i in range(len(mylist[0])):
    if mylist[0][i] != result[i]:
        count += 1
print(count / len(mylist[0]) * 100)
f = open("predictions_WR2021.txt", "w")

for line in result:
    f.write(line + '\n')
f.close()

exit()
