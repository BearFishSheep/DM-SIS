import numpy as np
import galois
from PIL import Image
import modify

# GF = galois.GF(2 ** 8)
GF = galois.GF(2 ** 16)


def get_E(M, GAS):
    E = []
    E1 = []  # 用于后面求广义逆
    for gas in GAS:
        temp1 = [GF(0) for _ in range(len(GAS))]
        temp = GF(temp1)
        for i in gas:
            temp1 = np.add(temp1, M[i - 1].tolist())
            temp = np.add(temp, GF(M[i - 1]))
        E.append(temp)
        E1.append(temp1)
    return E, E1


# 获得临时份额
def get_h(E_inv, s):
    return E_inv.dot(GF(s).reshape(len(E_inv), 1))


def get_shares(M, h):
    shares = []
    for i in range(len(M)):
        shares.append(GF(M[i]).reshape(1, len(M)).dot(h))
    return shares


def get_w(E, M, GAS):
    w = []
    for i in range(len(GAS)):
        e = E[i]
        m = []
        for j in range(len(GAS[i])):
            m.append(M[GAS[i][j] - 1])
        w.append(np.array(e).reshape(1, len(GAS)).dot(np.linalg.pinv(np.array(m))))  # 广义逆
    return w


# 这个方法适用于知道完整的E和M，这样求出来的是整个重构矩阵
# 并不像上面的方法一样是一个成员一个成员的求重构矩阵
# 如果是只恢复某个成员的秘密，且这个成员只知道它对应的e_j和M_Γ_j，那么还是用上面的方法
'''
def get_w1(E, M):
    w = []
    A = GF(E).dot(np.linalg.inv(GF(M)))
    for i in range(len(A)):
        temp = []
        for j in range(len(A[0])):
            if A[i][j] != 0:
                temp.append(A[i][j])
        w.append([temp])
    return w
'''


def recover(w, sh, GAS):
    s_recover = []
    for i in range(len(GAS)):
        sh_temp = []
        w_temp = w[i].flatten()
        w_temp_1 = []
        for j in range(len(GAS[i])):
            sh_temp.append(sh[GAS[i][j] - 1][0][0])     # 和shares得到的尺寸有关
            w_temp_1.append(int(round(w_temp[j], 0)))
        s_recover.append(int(np.dot(GF(w_temp_1), GF(sh_temp))))
    return s_recover


# 调整维度得到最后恢复的秘密
def get_recover_ultimate(s_recover, m, rands):
    s_recover_ultimate = [[] for _ in range(m)]
    s_recover = np.array(s_recover).flatten()  # 把初步恢复的秘密拉平
    for i in range(m):
        for j in range(i, len(s_recover), m):
            s_recover_ultimate[i].append(s_recover[j])
    s_recover_ultimate = reduce_random(s_recover_ultimate, rands)  # 去除随机数干扰项
    return s_recover_ultimate


# shares也要改变维度，因为我们要得到中间的影子图像
def get_shares_ultimate(shares, m):
    shares_ultimate = [[] for _ in range(m)]
    s_recover = np.array(shares).flatten()  # 把初步恢复的秘密拉平
    for i in range(m):
        for j in range(i, len(s_recover), m):
            shares_ultimate[i].append(s_recover[j])
    return shares_ultimate


# M得是可逆的，可以是范德蒙，也可以是三角矩阵，或者是带状矩阵
def get_M(n):
    M = []
    for i in range(n):
        temp = []
        for j in range(n):
            temp.append(GF(i + 1) ** j)
        M.append(GF(temp))
    return M


# 增加随机数干扰，增加安全性
def add_random(s, rands):
    for i in range(len(s)):
        for j in range(1, len(s[i])):
            s[i][j] = GF(s[i][j]) + rands[i] * GF(s[i][j - 1])
    return s


# 去除随机数，恢复原秘密
def reduce_random(s_recover, rands):
    for i in range(len(s_recover)):
        for j in range(1, len(s_recover[i])):
            s_recover[i][len(s_recover[i]) - j] \
                = int(GF(s_recover[i][len(s_recover[i]) - j]) -
                      rands[i] * GF(s_recover[i][len(s_recover[i]) - j - 1]))
    return s_recover


# 因为GF是2^16，所以要将两个像素合成为1个16位的数再进行操作
def two_to_one(s):
    s_temp = []
    for i in range(len(s)):
        temp = []
        for j in range(0, len(s[0]), 2):
            temp.append(int(bin(s[i][j] * 256 + s[i][j + 1]).replace('0b', '').zfill(16), 2))
        s_temp.append(temp)
    return s_temp


# 恢复的时候，要把16位的数据拆成2个8位
def one_to_two(s_recover):
    s_recover_temp = []
    for i in range(len(s_recover)):
        temp = []
        for j in range(len(s_recover[0])):
            x = bin(s_recover[i][j]).replace('0b', '').zfill(16)
            temp.append(int(x[: 8], 2))
            temp.append(int(x[8:], 2))
        s_recover_temp.append(temp)
    return s_recover_temp


def main():
    m = 5
    n = 4

    s1 = Image.open('./pic/Baboon.bmp').convert('L')
    s2 = Image.open('./pic/Barbara.bmp').convert('L')
    s3 = Image.open('./pic/Cameraman.bmp').convert('L')
    s4 = Image.open('./pic/Goldhill.bmp').convert('L')
    s5 = Image.open('./pic/Lena.bmp').convert('L')
    s6 = Image.open('./pic/Peppers.bmp').convert('L')
    # s1 = [[151, 72], [30, 45]]
    # s2 = [[237, 90], [87, 128]]
    # s3 = [[119, 160], [251, 255]]
    # s4 = [[19, 110], [220, 50]]
    # s5 = [[219, 210], [120, 150]]
    # s6 = [[119, 110], [170, 190]]
    # s1 = [[51]]
    # s2 = [[167]]
    # s3 = [[32]]
    # s4 = [[249]]
    # s5 = [[78]]
    # s6 = [[98]]
    shape = np.array(s1).shape
    s = [np.array(s1).flatten(), np.array(s2).flatten(),
         np.array(s3).flatten(), np.array(s4).flatten(),
         np.array(s5).flatten()]  # 秘密

    s = two_to_one(s)
    # print('s: ', s)

    # GAS = [[1, 2, 4], [1, 3], [2, 3], [1, 2, 3]]  # 访问结构3，m < n
    # GAS = [[1, 5], [2, 3], [1, 4], [1, 2, 3]]
    # GAS = [[1, 5], [2, 3], [1, 4], [1, 2, 3], [3, 4, 5]]
    GAS = [[3, 4], [2, 3], [1, 4], [1, 2, 3], [2, 3, 4]]
    GAS_matrix = modify.get_specific_matrix(GAS, len(GAS), n)
    print('GAS_matrix:', GAS_matrix)

    M = get_M(max(m, n))
    print('M:', M)

    # 判断是否需要进行调整
    subsets = modify.get_subsets(max(m, n))
    print('subsets:', subsets)
    if m == n:
        A = modify.get_A(GAS_matrix)
        print('A:', A)
        if np.linalg.det(GF(A)) == 0:
            u_matrix = modify.get_specific_matrix(subsets, len(subsets), max(m, n))
            v_matrix = modify.get_v_matrix(u_matrix)
            C = modify.modify1(GAS_matrix, u_matrix, v_matrix)
            A = modify.get_A(C)
            M = get_M(n + 1)
            print('A:', A)
            GAS = modify.get_GAS_modify(C)
            print('GAS_modify:', GAS)
    else:
        if m < n:
            for _ in range(n):
                subsets.pop(0)
        u_matrix = modify.get_specific_matrix(subsets, len(subsets), max(m, n))
        print('u_matrix:', u_matrix)
        if m > n:
            C = modify.modify2(GAS_matrix, u_matrix, m - n)
        else:
            C = modify.modify3(GAS_matrix, u_matrix, n - m)
        A = modify.get_A(C)
        print('A:', A)
        GAS = modify.get_GAS_modify(C)
        print('GAS_modify:', GAS)

    E = GF(A).dot(GF(M))
    E1 = np.array(A).dot(np.array(M))
    print('E:', E)
    # print('E1:', E1)
    # print('det(E):', np.linalg.det(E))
    # print('det(GF(E)):', np.linalg.det(GF(E)))

    E_inv = np.linalg.inv(GF(E))
    print('E_inv:', E_inv)

    rands = [GF(np.random.randint(1, 65536)) for _ in range(max(max(m, n), len(M)))]  # 增加随机数干扰项，提高安全性
    print('rands:', rands)

    # 如果m < n或len(M) > n，要给s添加随机数，显式地添加
    if m < n or (m == n and len(M) > n):
        if m < n:
            num = n - m
        else:
            num = len(M) - n
        for _ in range(num):
            # s.append([np.random.randint(256) for _ in range(len(s[0]))])
            s.append([np.random.randint(65536) for _ in range(len(s[0]))])
    # print('s:', s)

    s = add_random(s, rands)
    # print('增加随机数干扰后:', s)

    h = []
    for i in range(len(s[0])):  # 每个秘密已经拉平了，都是一维
        s_temp = []
        for j in range(len(s)):
            s_temp.append(s[j][i])
        h.append(get_h(E_inv, s_temp))
    # print('h:', h)

    shares = []
    for i in range(len(h)):
        shares.append(get_shares(M, h[i]))
    # print('shares:', shares)

    shares2 = get_shares_ultimate(shares, max(max(m, n), len(M)))
    shares2 = one_to_two(shares2)
    # print('shares2:', shares2)
    to_save = np.array(shares2)
    # for i in range(len(to_save)):
    #     Image.fromarray(to_save[i].reshape(shape).astype(np.uint8)) \
    #         .save("./sis_16/mGTn(3-8)/share_{}.jpg".format(i + 1))

    # 如果m>n或m<n进行了调整，要对GAS进行更新
    w = get_w(E1, M, GAS)
    # print('w:', w)

    s_recover = []
    for i in range(len(shares)):
        s_recover.append(recover(w, shares[i], GAS))
    # print('s_recover:', s_recover)

    s_recover_ultimate = get_recover_ultimate(s_recover, max(max(m, n), len(M)), rands)
    s_recover_ultimate = s_recover_ultimate[: m]  # m<n时，只需要取前m个秘密
    s_recover_ultimate = one_to_two(s_recover_ultimate)     # 从16位变为2个8位
    print('s_recover_ultimate:', s_recover_ultimate)
    to_save2 = np.array(s_recover_ultimate)
    # for i in range(len(to_save2)):
    #     Image.fromarray(to_save2[i].reshape(shape).astype(np.uint8)) \
    #         .save("./sis_16/mGTn(3-8)/reconstruct_{}.jpg".format(i + 1))


if __name__ == '__main__':
    main()

