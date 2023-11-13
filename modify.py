import itertools
import numpy as np
import galois

GF = galois.GF(2 ** 8)
# GF = galois.GF(2 ** 16)
# GF = galois.GF(2 ** 32)


# 获得子集向量，为后面生成u_matrix和v_matrix做铺垫
def get_subsets(n):
    nums = [i + 1 for i in range(n)]
    res = []

    for i in range(len(nums) + 1):
        for tmp in itertools.combinations(nums, i):
            res.append(tmp)
    res.pop(0)

    return res


# 获得特定的矩阵（初始指示矩阵、访问结构对应的初始重构矩阵）
def get_specific_matrix(matrix, width, height):
    specific_matrix = np.zeros((width, height))

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            specific_matrix[i][matrix[i][j] - 1] = 1

    return specific_matrix.astype(np.uint8)


'''
# 获得初始指示矩阵
def generate_matrix(n, subsets):
    matrix = np.zeros((len(subsets), n))
    for i in range(len(subsets)):
        for j in range(len(subsets[i])):
            matrix[i][subsets[i][j] - 1] = 1
    return matrix.astype(np.uint8)


# 获得访问结构对应的初始重构矩阵
def get_GAS_matrix(GAS):
    GAS_matrix = np.zeros((len(GAS), len(GAS)))
    for i in range(len(GAS)):
        for j in range(len(GAS[i])):
            GAS_matrix[i][GAS[i][j] - 1] = 1
    return GAS_matrix.astype(np.uint8)
'''


# 加一列全1
def get_v_matrix(u_matrix):
    v_temp = np.ones(len(u_matrix)).reshape(len(u_matrix), 1)  # 全1
    return np.concatenate((u_matrix, v_temp), axis=1).astype(np.uint8)


# 调整策略1：加一行和一列
def modify1(GAS_matrix, u_matrix, v_matrix):
    C = np.array([])
    index = []  # 记录此时选中了哪一行哪一列

    for i in range(len(u_matrix)):
        if u_matrix[i].tolist() not in GAS_matrix.tolist():
            C = np.concatenate((GAS_matrix, u_matrix[i].reshape(1, len(u_matrix[i]))), axis=0)
            index.append(i)
            temp = C

            for j in range(len(v_matrix)):
                if v_matrix[j].tolist() not in C.T.tolist():
                    if len(index) > 1:
                        index.pop()
                    C = np.concatenate((temp, v_matrix[j].reshape(len(v_matrix[j]), 1)), axis=1)
                    index.append(j)
                    if np.linalg.det(GF(get_A(C))) != 0:
                        break

            if np.linalg.det(GF(get_A(C))) != 0:
                break
            else:
                if len(index) > 0:
                    index.pop()  # 否则就删除里面两个元素
                if len(index) > 0:
                    index.pop()

    return C


# 调整策略2：加m - n列，num是m - n
def modify2(GAS_matrix, u_matrix, num):
    index = []  # 不在gas_matrix矩阵中的m-n个索引号码的组合
    C = np.array([])

    for i in range(len(u_matrix)):
        if u_matrix[i].tolist() not in GAS_matrix.T.tolist():
            index.append(i)
    index = reversed(index)

    # combinations = list(itertools.combinations(index, k))
    combinations = lazy_combinations(index, num)  # 懒加载生成器

    try:
        while True:
            C = GAS_matrix
            combination = combinations.__next__()
            for i in combination:
                C = np.concatenate((C, u_matrix[i].reshape(len(u_matrix[i]), 1)), axis=1)
            if np.linalg.det(GF(get_A(C))) != 0:
                break
    except StopIteration:
        pass

    # for combination in combinations:
    #     C = GAS_matrix
    #     for i in combination:
    #         C = np.concatenate((C, u_matrix[i].reshape(len(u_matrix[i]), 1)), axis=1)
    #     if np.linalg.det(GF(get_A(C))) != 0:
    #         break

    return C


# 懒加载，每次只返回一个，提高组合的效率
def lazy_combinations(iterable, r):
    # 定义生成器函数
    for c in itertools.combinations(iterable, r):
        yield c


def modify_kn(m, num, GAS_matrix):
    count = 1
    arr = [i + 1 for i in range(m)]
    combination = [[1 for _ in range(m)]]

    c1 = list(itertools.combinations(arr, m - 1))
    c1_matrix = get_specific_matrix(c1, len(c1), m)

    # if len(c1_matrix) + count < num:
    #     print('------------------------------------')
    #     combination += c1_matrix.tolist()
    #     c2 = list(itertools.combinations(arr, m - 2))
    #     c2_matrix = get_specific_matrix(c2, len(c2), m)
    #     combination += c2_matrix[: len(c2_matrix) - len(c1_matrix) - count].tolist()

    combination += c1_matrix.tolist()
    combination = combination[: num]

    C = GAS_matrix
    for comb in combination:
        C = np.concatenate((C, np.array(comb).reshape(len(np.array(comb)), 1)), axis=1)

    return C


# 调整策略3：加n - m行，num是n - m
def modify3(GAS_matrix, u_matrix, num):
    index = []  # 不在gas_matrix矩阵中的m-n个索引号码的组合
    C = np.array([])

    for i in range(len(u_matrix)):
        if u_matrix[i].tolist() not in GAS_matrix.tolist():
            index.append(i)
    index = reversed(index)

    # combinations = list(itertools.combinations(index, k))
    combinations = lazy_combinations(index, num)

    try:
        while True:
            C = GAS_matrix
            combination = combinations.__next__()
            for i in combination:
                C = np.concatenate((C, u_matrix[i].reshape(1, len(u_matrix[i]))), axis=0)
            if np.linalg.det(GF(get_A(C))) != 0:
                break
    except StopIteration:
        pass

    # for combination in combinations:
    #     C = GAS_matrix
    #     for i in combination:
    #         C = np.concatenate((C, u_matrix[i].reshape(1, len(u_matrix[i]))), axis=0)
    #     if np.linalg.det(GF(get_A(C))) != 0:
    #         break

    return C


# 重构矩阵A
def get_A(C):
    A = np.zeros(C.shape)
    for i in range(len(C)):
        for j in range(len(C[i])):
            if C[i][j] == 1:
                A[i][j] = GF(np.random.randint(1, 256))
                # A[i][j] = GF(np.random.randint(1, 65536))   # GF(2^16)的情况
                # A[i][j] = GF(np.random.randint(1, 65536 ** 2, dtype=np.int64))   # GF(2^32)的情况
    return A.astype(int)  # 不能写成np.int8
    # return A.astype(np.int64)


# 得到调整后的GAS
def get_GAS_modify(C):
    GAS = []
    for i in range(len(C)):
        temp = []
        for j in range(len(C[i])):
            if C[i][j] == 1:
                temp.append(j + 1)
        GAS.append(temp)
    return GAS


