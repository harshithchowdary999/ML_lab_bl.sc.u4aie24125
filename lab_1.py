# 1. Count pairs with sum = 10
def countofpairwithsumis10(list1):
    # count unique pairs with sum 10
    count=0
    for i in range(len(list1)):
        for j in range(i+1,len(list1)):
            if list1[i]+list1[j]==10:
                count+=1
    return count
# 2. Range of list of real numbers
def listofrealnumberswithrange():
    l=[]
    try:
        n = int(input("Enter size of list: "))
        if n<=3:
            raise ValueError

    except ValueError:
        print("entry must be greater than 3")
    for i in range(n):
        k= int(input("Enter  element "))
        l.append(k)

    return max(l)-min(l)
# 3. Matrix power
import numpy as np
from numpy.linalg import matrix_power
#from scipy documentation

def matrixoperation(A,m):

    while m<=0:
        print("it should be a positive integer")
        m = int(input("Enter power"))
    ans = matrix_power(A,m)
    return ans
# 4. Highest occurring alphabet character
def highestcountofchar(string1):
    k=set(string1)
    list1=list(k)
    list2=[]
    for i in list1:
        g=string1.count(i)
        list2.append(g)
    d1=dict(zip(list2,list1))
    m1=max(list2)
    return d1[m1],str(m1)

# 5. Mean, Median, Mode of random numbers
import random
import statistics
#from python.org
def statsticss():
    listsize = 25
    min1=1
    max1=10
    random_integer=[random.randint(min1,max1) for i in range(listsize)]
    random_integer.sort()
    mean=sum(random_integer)/listsize
    median=statistics.median(random_integer)
    mode=statistics.mode(random_integer)
    return random_integer,mean,median,mode
def main():
    # 1
    list1 = [2, 7, 4, 1, 3, 6]
    print("1. Pairs with sum 10:", countofpairwithsumis10(list1))

    # 2

    print("2. Range of list:", listofrealnumberswithrange())

    # 3
    A = np.array([[1, 2], [3, 4]])
    m = 2
    print("3. Matrix A^m:\n", matrixoperation(A, m))

    # 4
    s = "hippopotamus"
    char, count = highestcountofchar(s)
    print(f"4. Highest occurring character: '{char}' occurs {count} times")

    # 5
    nums, mean, median, mode = statsticss()
    print("5. Random Numbers:", nums)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)



if __name__ == "__main__":
    main()




