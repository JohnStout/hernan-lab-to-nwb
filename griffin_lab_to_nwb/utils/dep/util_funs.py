def find(value,listVar):
    index = []
    counter = 0
    for i in range(len(listVar)):
        if value in listVar[i]:
            index.append(i)
    return index
        