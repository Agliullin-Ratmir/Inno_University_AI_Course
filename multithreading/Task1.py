from concurrent.futures import ThreadPoolExecutor

def isSimple(n):
    if n == 1 or n == 2 or n == 3:
        return True
    if n == 0:
        return False
    for i in range(2, n):
        if n%i == 0:
            return False
    return True

def count_simples(interval):
    start, end = interval
    count = 0
    for i in range(start, end+1):
        if isSimple(i):
            print(f"{i} is simple")
            count += 1
        else:
            print(f"{i} is not simple")
    return count

def get_intervals(n): #3 intervals
    step = n // 3
    ranges = [(1, step), (step + 1, 2 * step), ((2*step) + 1, n)]
    print(ranges)
    return ranges

if __name__ == "__main__":
    ranges = get_intervals(10)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(count_simples, ranges)
    print(sum(results))



