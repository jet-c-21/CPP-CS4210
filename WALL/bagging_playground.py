def instance_be_chosen(n: int):
    return 1 - ((1 - (1 / n)) ** n)


for i in [10, 100, 1000, 10000, 100000, 1000000]:
    p = instance_be_chosen(i)
    print(f"n = {str(i).ljust(10)},p = {str(p).ljust(10)}")
