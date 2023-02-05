# # 1 - Bubble sort , Selection sort , Insertion sort , Merge sort , Quick sort
#
# import random
# import timeit
# list_of_numbers_1 = [random.randint(1, 101) for i in range(100)]
# list_of_numbers_2 = [random.randint(1, 1001) for j in range(1000)]
# list_of_numbers_3 = [random.randint(1, 10001) for k in range(10000)]
# def bubble_sort(nums:list):
#     startTime = timeit.default_timer()
#     swapped = True
#     while swapped:
#         swapped = False
#         for i in range(len(nums) - 1):
#             if nums[i] > nums[i + 1]:
#                 nums[i],nums[i+1] = nums [i+1],nums[i]
#                 swapped = True
#     elapsedTime = (timeit.default_timer()-startTime)
#     print('Time taken to start',elapsedTime)
# def selection_sort(nums: list):
#     startTime = timeit.default_timer()
#     for i in range(len(nums)):
#         lowest_value_index = i
#         for j in range(i + 1, len(nums)):
#             if nums[j] < nums[lowest_value_index]:
#                 lowest_value_index = j
#         nums[i], nums[lowest_value_index] = nums[lowest_value_index],nums[i]
#     elapsedTime = (timeit.default_timer() - startTime)
#     print('Time taken to sort: ', elapsedTime)
# def insertion_sort(nums:list):
#    startTime = timeit.default_timer()
#    for i in range(1, len(nums)):
#       item_to_insert = nums[i]
#       j = i - 1
#       while j >= 0 and nums[j] > item_to_insert:
#           nums[j + 1] = nums[j]
#           j -= 1
#       nums[j + 1] = item_to_insert
#    elapsedTime = (timeit.default_timer() - startTime)
#    print('Time taken to sort: ', elapsedTime)
# def merge_sort(nums:list):
#     startTime = timeit.default_timer()
#     if len(nums) > 1:
#         left_nums = nums[:len(nums) // 2]
#         right_nums = nums[len(nums) // 2:]
#         merge_sort(left_nums)
#         merge_sort(right_nums)
#         i = 0
#         j = 0
#         k = 0
#         while i < len(left_nums) and j < len(right_nums):
#             if left_nums[i] < right_nums[j]:
#                nums[k] = left_nums[i]
#                i += 1
#             else:
#                nums[k] = right_nums[j]
#                j += 1
#             k += 1
#         while i < len(left_nums):
#             nums[k] = left_nums[i]
#             i += 1
#             k += 1
#         while j < len(right_nums):
#             nums[k] = right_nums[i]
#             j += 1
#             k += 1
#     elapsedTime = (timeit.default_timer() - startTime)
#     print('Time taken to sort: ', elapsedTime)
# def merge_sort(nums: list):
#      if len(nums) > 1:
#         left_nums = nums[:len(nums) // 2]
#         right_nums = nums[len(nums) // 2:]
#         merge_sort(left_nums)
#         merge_sort(right_nums)
#         i = 0
#         j = 0
#         k = 0
#         while i < len(left_nums) and j < len(right_nums):
#             if left_nums[i] < right_nums[j]:
#                 nums[k] = left_nums[i]
#                 i += 1
#             else:
#                 nums[k] = right_nums[j]
#                 j += 1
#             k = + 1
#         while i < len(left_nums):
#             nums[k] = left_nums[i]
#             i += 1
#             k += 1
#         while j < len(right_nums):
#             nums[k] = right_nums[j]
#             j += 1
#             k += 1
# def quick_sort(nums:list,left,right):
#     if left < right:
#         partition_pos = partition(nums, left, right)
#         quick_sort(nums,left,partition_pos - 1)
#         quick_sort(nums, partition_pos + 1, right)
# def partition(nums:list, left, right):
#     i = left
#     j = right - 1
#     pivot = nums[right]
#     while i < j:
#        while i < right and nums[i] < pivot:
#           i += 1
#        while j > left and nums[j] >= pivot:
#           j -= 1
#        if i < j:
#           nums[i], nums[j] = nums[j], nums[i]
#     if nums[i] > pivot:
#        nums[i], nums[right] = nums[right], nums[i]
#     return i
# print('\n************************1.Bubble Sorted Lists**************************')
# print('Time Complexity of First List with 100 random numbers')
# bubble_sort(list_of_numbers_1)
# print('Time Complexity of Second List with 1000 random numbers')
# bubble_sort(list_of_numbers_2)
# print('Time Complexity of Third List with 10000 random numbers')
# bubble_sort(list_of_numbers_3)
# print("\n")


# 2 - Searching algoritham

# import timeit
# def linear_search(a, x):
#     for i in range(len(a)):
#         if a[i] == x:
#             return i
#     return -1
# def binary_search(array, low, high, x):
#     if high >= low:
#         mid = (high + low) // 2
#         if array[mid] == x:
#             return mid
#         elif array[mid] > x:
#             return binary_search(array, low, mid - 1, x)
#         else:
#             return binary_search(array, mid + 1, high, x)
#     else:
#         return -1
# array = []
# num = int(input("Enter the number of element : "))
# for i in range(0, num):
#     print("ELement no. ", i + 1, " - ", end=' ')
#     ele = int(input())
#     array.append(ele)
# print("\n Element list is ", array)
# q = 1
# while q == 1:
#     print("\nPlease choose searching algoritham in which you want to search!!")
#     print("\nPress 1 for LINEAR SEARCH..")
#     print("Press 2 for BINARY SEARCH..")
#     n = int(input("Your choice : "))
#     if n == 1 or n == 2:
#         print("\nPlease Enter your number which you want to search..")
#         a = int(input())
#         for i in range (0,len(array)-1) :
#             if array[i]==a:
#                 break
#             else :
#                 continue
#         if a!=array[i]:
#             print("Your number is not in list")
#         else:
#             if n == 1:
#                 startTime = timeit.default_timer()
#                 z = linear_search(array, a)
#                 elapsedTime = (timeit.default_timer() - startTime)
#                 print(z, "Index No. in list")
#                 print('Time taken to search: ', elapsedTime)
#             else:
#                 startTime = timeit.default_timer()
#                 z = binary_search(array, 0, len(array) - 1, a)
#                 elapsedTime = (timeit.default_timer() - startTime)
#                 print(z, "Index No. in list")
#                 print('Time taken to search: ', elapsedTime)
#     else:
#         print("Please Enter Valid input!!!")
#     print("Do you want to search again in that list (y/n) ??")
#     n1 = str(input())
#     if n1 == 'y':
#         q = 1
#     elif n1 == 'n':
#         q = 0
#     else:
#         print("You Entered wrong string..\n Exiting the program..")
#         q = 0


# # 3 - RECURSIVE CODE TO FIND FACTORIAL OF NUMBER
# def recursive_factorial(n):
#     if n==1:
#         return n
#     else:
#         return n*recursive_factorial(n-1)
# #ITERATIVE CODE TO FIND FACTORIAL OF NUMBER
# def iterative_factorial(n):
#     if n<0:
#         return 0
#     elif n == 0 or n==1:
#         return 1
#     else:
#         fact=1
#         while(n>1):
#             fact *= n
#             n -= 1
#         return fact
# import timeit
# print("Please Enter your number : ", end=' ')
# n=int(input())
# print("\n***************  FACTORIAL USING RECURSIVE METHOD  ******************\n")
# startTime = timeit.default_timer()
# z=recursive_factorial(n)
# print("FACTORIAL = ",z)
# elapsedTime = (timeit.default_timer() - startTime)
# print('Time taken for recursive method : ', elapsedTime)
# print("\n***************  FACTORIAL USING ITERATIVE METHOD  ******************\n")
# startTime = timeit.default_timer()
# z=iterative_factorial(n)
# print("FACTORIAL = ",z)
# elapsedTime = (timeit.default_timer() - startTime)
# print('Time taken for iterative method : ', elapsedTime)
# print("\n*********************************************************************\n")


# # 4 - Fractional Knapsack problem
# def fractional_knapsack(value,weight,capacity):
#     index=list(range(len(value)))
#     ratio=[v/w for v,w in zip(value,weight)]
#     index.sort(key=lambda i: ratio[i], reverse=True)
#     max_value=0
#     fraction = [0] * len(value)
#     for i in index:
#         if weight[i]<= capacity:
#             fraction[i]=1
#             max_value += value[i]
#             capacity -= weight[i]
#         else:
#             fraction[i] = capacity / weight[i]
#             max_value += value[i]*capacity/weight[i]
#             break
#     return max_value,fraction
# n = int(input("Enter number of items : "))
# value = input(f"Enter the values of the {n} item(s) in order : ").split()#.formate()
# value = [int(v) for v in value]
# weight = input("Enter the positive weights of the {} items(s) in order : ".format(n)).split()
# weight = [int(w) for w in weight]
# capacity = int(input("Enter maximum weight: "))
# max_value, fractions = fractional_knapsack(value, weight, capacity)
# print("The maximum value of items that can be carried : ", max_value)
# print("The fraction in which the items should be taken : ", fractions)


# # 5 - Prim's Minimum Spanning Tree (MST) algorithm
# import sys  # Library for INT_MAX
# class Graph():
#     def __init__(self, vertices):
#         self.V = vertices
#         self.graph = [[0 for column in range(vertices)]
#                          for row in range(vertices)]
#     def printMST(self, parent):
#         print(" Edge \tWeight")
#         for i in range(1, self.V):
#             print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
#     def minKey(self, key, mstSet):
#         min = sys.maxsize
#         for v in range(self.V):
#             if key[v] < min and mstSet[v] == False:
#                 min = key[v]
#                 min_index = v
#         return min_index
#     def primMST(self):
#         key = [sys.maxsize] * self.V
#         parent = [0] * self.V
#         key[0] = 0
#         mstSet = [0] * self.V
#         parent[0] = -1  # First node is always the root of
#         for cout in range(self.V):
#             u = self.minKey(key, mstSet)
#             mstSet[u] = True
#             for v in range(self.V):
#                 if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
#                     key[v] = self.graph[u][v]
#                     parent[v] = u
#         self.printMST(parent)
# g = Graph(5)
# g.graph = [[0, 2, 0, 6, 0],
#            [2, 0, 3, 8, 5],
#            [0, 3, 0, 0, 7],
#            [6, 8, 0, 0, 9],
#            [0, 5, 7, 9, 0]]
# g.primMST()

# # 6 - Kruskal's algorithm
# class Graph:
#     def __init__(self, vertices):
#         self.V = vertices
#         self.graph = []
#     def addEdge(self, u, v, w):
#         self.graph.append([u, v, w])
#     def find(self, parent, i):
#         if parent[i] == i:
#             return i
#         return self.find(parent, parent[i])
#     def union(self, parent, rank, x, y):
#         xroot = self.find(parent, x)
#         yroot = self.find(parent, y)
#         if rank[xroot] < rank[yroot]:
#             parent[xroot] = yroot
#         elif rank[xroot] > rank[yroot]:
#             parent[yroot] = xroot
#         else:
#             parent[yroot] = xroot
#             rank[xroot] += 1
#     def KruskalMST(self):
#         result = []
#         i = 0
#         e = 0
#         self.graph = sorted(self.graph, key=lambda item: item[2])
#         parent = []
#         rank = []
#         for node in range(self.V):
#             parent.append(node)
#             rank.append(0)
#         while e < self.V - 1:
#             u, v, w = self.graph[i]
#             i = i + 1
#             x = self.find(parent, u)
#             y = self.find(parent, v)
#             if x != y:
#                 e = e + 1
#                 result.append([u, v, w])
#                 self.union(parent, rank, x, y)
#         minimumCost = 0
#         print("Edges in the constructed MST")
#         for u, v, weight in result:
#             minimumCost += weight
#             print("%d -- %d == %d" % (u, v, weight))
#         print("Minimum Spanning Tree", minimumCost)
# g = Graph(4)
# g.addEdge(0, 1, 10)
# g.addEdge(0, 2, 6)
# g.addEdge(0, 3, 5)
# g.addEdge(1, 3, 15)
# g.addEdge(2, 3, 4)
# g.KruskalMST()


# # 7 - Heapsort
# def heapify(arr, n, i):
#     largest = i
#     l = 2 * i + 1
#     r = 2 * i + 2
#     if l < n and arr[i] < arr[l]:
#         largest = l
#     if r < n and arr[largest] < arr[r]:
#         largest = r
#     if largest != i:
#         arr[i], arr[largest] = arr[largest], arr[i]
#         heapify(arr, n, largest)
# def heapSort(arr):
#     n = len(arr)
#     for i in range(n // 2 - 1, -1, -1):
#         heapify(arr, n, i)
#     for i in range(n - 1, 0, -1):
#         arr[i], arr[0] = arr[0], arr[i]
#         heapify(arr, i, 0)
# arr = [12, 11, 13, 5, 6, 7]
# heapSort(arr)
# n = len(arr)
# print("Sorted array is")
# for i in range(n):
#     print(arr[i],end=' ')


# 8 - LCS using dynamic programming

# def lcs(x, y):
#     m = len(x)
#     n = len(y)
#     L = [[0] * (n + 1) for i in range(m + 1)]
#     for i in range(m + 1):
#         for j in range(n + 1):
#             if i == 0 or j == 0:
#                 L[i][j] = 0
#             elif x[i - 1] == y[j - 1]:
#                 L[i][j] = L[i - 1][j - 1] + 1
#             else:
#                 L[i][j] = max(L[i - 1][j], L[i][j - 1])
#     return L[m][n]
# X = "AGGTAB"
# Y = "GXTXAYB"
# print("Length of lCS is ", lcs(X, Y))


# 9 - Knapsack using dynamic programming

# def knapsack(capacity, wt, val, n):
#     K = [[0 for x in range(capacity + 1)]
#          for x in range(n + 1)]
#     for i in range(n + 1):
#         for j in range(capacity + 1):
#             if i == 0 or j == 0:
#                 K[i][j] = 0
#             elif wt[i - 1] <= j:
#                 K[i][j] = max(val[i - 1] + K[i - 1][j - wt[i - 1]], K[i - 1][j])
#             else:
#                 K[i][j] = K[i - 1][j]
#     return K[n][capacity]
# val = [60, 100, 120]
# wt = [10, 20, 30]
# capacity = 50
# n = len(val)
# print("Total Value = ", knapsack(capacity, wt, val, n))


# # 10 - Making Change Coin Problem using Dynamic Programming
#
# import numpy as np
# from prettytable import PrettyTable
# def Making_Change_Problem(coins, change):
#     a = np.zeros((len(coins), change + 1), dtype=int)
#     n, l = len(coins), list(range(0, len(coins)))
#     table = PrettyTable([k - 1 if k - 1 != -1 else " " for k in range(change + 2)])
#     for i in range(n):
#         for j in range(1, (change + 1)):
#             if i == 0:
#                 a[i][j] = j // coins[i]
#             elif j < coins[i]:
#                 a[i][j] = a[i - 1][j]
#             elif j >= coins[i]:
#                 a[i][j] = min(a[i - 1][j], ((j // coins[i]) + a[i][j % coins[i]]))
#     for i in range(n):
#         l1 = []
#         for j in range(-1, change + 1):
#             if j == -1:
#                 l1.append(coins[i])
#             else:
#                 l1.append(a[i][j])
#         table.add_row([i for i in l1])
#     print(table)
#     total_coins_req = a[i][j]
#     coins_req = []
#     #Printing Sequence
#     while i >= 0 and j >= 0 and change > 0:
#         while a[i][j] == a[i - 1][j] and i - 1 >= 0:
#             i -= 1
#         coins_req.append(coins[i])
#         j = change - coins[i]
#         change -= coins[i]
#     return total_coins_req, coins_req
# coin = list(map(int, input("Enter coins (space separated): ").split()))
# price = int(input("Enter Change you want to make: "))
# a,b=Making_Change_Problem(coin,price)
# print("Total Coins = ",a)
# print("Sequence = ",b)


# 11 - Matrix Multiplication Problem using Dynamic Programming
# import sys
# # def printParenthesis(m, j, i):
# #     if j == i:
# #         print(chr(65 + j), end="")
# #         return
# #     else:
# #         print("(", end="")
# #         printParenthesis(m, int(m[j][i]) - 1, i)
# #         printParenthesis(m, j, int(m[j][i]))
# #         print(")", end="")
# def MatrixChainOrder(array,n):
#     m=[[0 for x in range(n)]
#           for x in range(n)]
#     for i in range (1,n):
#         m[i][i]=0
#     for L in range (2,n):
#         for i in range (1,n-L+1):
#             j=i+L-1
#             m[i][j]=sys.maxsize
#             for k in range(i,j):
#                 q=m[i][k]+m[k+1][j]+array[i-1]*array[k]*array[j]
#                 if q<m[i][j]:
#                     m[i][j]=q
#     print("Minimum number of multiplications is ",m[1][n-1])
#     print("Optimal Parenthesis of is: ", end="")
#     # printParenthesis(m, n - 2, 0)
# arr = [1, 2, 3, 4]
# size = len(arr)
# MatrixChainOrder(arr,size)


## 12 DFS and BFS
# #DFS
# from collections import defaultdict
# class Graph():
#     def __init__(self):
#         self.graph=defaultdict(list)
#     def addEdge(self,u,v):
#         self.graph[u].append(v)
#     def DFSUtil(self,v,visited):
#         visited.add(v)
#         print(v,end=' -> ')
#         for neighbour in self.graph[v]:
#             if neighbour not in visited:
#                 self.DFSUtil(neighbour,visited)
#     def DFS(self,v):
#         visited=set()
#         self.DFSUtil(v,visited)
#     def BFS(self,s):
#         visited=[0]*(max(self.graph)+1)
#         queue=[]
#         queue.append(s)
#         visited[s]=True
#         while queue:
#             s=queue.pop(0)
#             print(s,end=' -> ')
#             for i in self.graph[s]:
#                 if visited[i]==False:
#                     queue.append(i)
#                     visited[i]=True
# g=Graph()
# g.addEdge(0,1)
# g.addEdge(0,2)
# g.addEdge(1,2)
# g.addEdge(2,0)
# g.addEdge(2,3)
# g.addEdge(3,3)
# p=int(input("Enter Source Vertex : "))
# print("(BFS) ")
# g.DFS(p)
# print("\n(DFS) ")
# g.BFS(p)

# # 13 - KMP Algorithm
# def KMPSearch(pat,txt):
#     M= len(pat)
#     N= len(txt)
#     lps=[0]*M
#     j=0
#     computeLPSArray(pat,M,lps)
#     i=0
#     while i<N:
#         if pat[j] == txt[i]:
#             i +=1
#             j +=1
#         if j==M:
#             print("Found pattern at idex " + str(i-j))
#             j = lps[j-1]
#         elif i<N and pat[j]!= txt[i]:
#             if j!=0:
#                 j = lps[j-1]
#             else:
#                 i+=1
# def computeLPSArray(pat, M, lps):
#     len = 0
#     lps[0]
#     i = 1
#     while i < M:
#         if pat[i] == pat[len]:
#             len += 1
#             lps[i] = len
#             i += 1
#         else:
#             if len != 0:
#                 len = lps[len - 1]
#             else:
#                 lps[i] = 0
#                 i += 1
# txt = "ABABDABACDABABCABAB"
# pat = "ABABCABAB"
# KMPSearch(pat, txt)
#
# # 14 - N Queen Problem
# global N
# N = 4
# def printSolution(board):
#     for i in range(N):
#         for j in range(N):
#             print(board[i][j], end=" ")
#         print()
# def isSafe(board, row, col):
#     for i in range(col):
#         if board[row][i] == 1:
#             return False
#     for i, j in zip(range(row, -1, -1),
#                     range(col, -1, -1)):
#         if board[i][j] == 1:
#             return False
#     for i, j in zip(range(row, N, 1),
#                     range(col, -1, -1)):
#         if board[i][j] == 1:
#             return False
#     return True
# def solveNQUtil(board, col):
#     if col >= N:
#         return True
#     for i in range(N):
#         if isSafe(board, i, col):
#             board[i][col] = 1
#             if solveNQUtil(board, col + 1) == True:
#                 return True
#             board[i][col] = 0
#     return False
# def solveNQ():
#     board = [[0, 0, 0, 0],
#              [0, 0, 0, 0],
#              [0, 0, 0, 0],
#              [0, 0, 0, 0]]
#     if solveNQUtil(board, 0) == False:
#         print("Solution does not exist")
#         return False
#     printSolution(board)
#     return True
# solveNQ()
