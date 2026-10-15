def swap(arr, a, b):
    t = arr[a]
    arr[a] = arr[b]
    arr[b] = t

def cocktailsort(order):
    dir = 1
    i = 1
    s = False
    changed = False
    while s is False:
        if i < 1 or i >= len(order) - 1:
            dir = dir * -1
            if changed == False:
                s = True
            else:
                changed = False
        if dir == 1:
            if order[i + 1] < order[i]:
                swap(order, i, i + 1)
                print(order)
                changed = True
                yield
            i += 1
        else:
            if order[i - 1] > order[i]:
                swap(order, i, i - 1)
                print(order)
                changed = True
                yield
            i -= 1

order = [8, 9, 1, 2, 5, 4, 8, 9, 0, 1, -2]
print(order)
cocktailsort(order)
print(order)
