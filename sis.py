import numpy as np
import galois
import modify

GF = galois.GF(2 ** 8)


# 获得目标矩阵
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


# 获得共享份额
def get_shares(M, h):
    shares = []
    for i in range(len(M)):
        shares.append(GF(M[i]).reshape(1, len(M)).dot(h))
    return shares


# 获得重构向量
def get_w(E, M, GAS):
    w = []
    
    for i in range(len(GAS)):
        e = E[i]
        m = []
        for j in range(len(GAS[i])):
            m.append(M[GAS[i][j] - 1])
        w.append(np.array(e).reshape(1, len(GAS)).dot(np.linalg.pinv(np.array(m))))  # 广义逆
    
    return w


'''
# 这个方法适用于知道完整的E和M，这样求出来的是整个重构矩阵
# 并不像上面的方法一样是一个成员一个成员的求重构矩阵
# 如果是只恢复某个成员的秘密，且这个成员只知道它对应的e_j和M_Γ_j，那么还是用上面的方法
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


# 恢复秘密
def recover(w, sh, GAS):
    s_recover = []
    
    for i in range(len(GAS)):
        sh_temp = []
        w_temp = w[i].flatten()
        w_temp_1 = []
        for j in range(len(GAS[i])):
            sh_temp.append(sh[GAS[i][j] - 1][0][0])  # 和shares得到的尺寸有关
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



