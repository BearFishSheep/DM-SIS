from sis import *
from PIL import Image


def main():
    m = 4
    n = 5

    s1 = Image.open('./pic/Baboon.bmp').convert('L')
    s2 = Image.open('./pic/Barbara.bmp').convert('L')
    s3 = Image.open('./pic/Cameraman.bmp').convert('L')
    s4 = Image.open('./pic/Goldhill.bmp').convert('L')
    s5 = Image.open('./pic/Lena.bmp').convert('L')
    s6 = Image.open('./pic/Peppers.bmp').convert('L')

    shape = np.array(s1).shape
    s = [np.array(s1).flatten(), np.array(s2).flatten(),
         np.array(s3).flatten(), np.array(s4).flatten()]  # 秘密，m = n
    # s = [np.array(s1).flatten(), np.array(s2).flatten(),
    #      np.array(s3).flatten(), np.array(s4).flatten(),
    #      np.array(s5).flatten(), np.array(s6).flatten()]  # 秘密，m > n
    s = [np.array(s1).flatten(), np.array(s2).flatten(),
         np.array(s3).flatten(), np.array(s4).flatten()]  # 秘密，m < n

    # GAS = [[1, 2], [3, 4], [2, 3, 4], [1, 2, 3]]  # 访问结构1，m = n
    # GAS = [[1, 3, 4], [1, 2], [2, 3, 4], [1, 2, 4], [1, 3], [1, 2, 3]]   # 访问结构2，m > n
    GAS = [[1, 4, 5], [2, 4], [1, 3], [3, 5]]       # 访问结构3，m < n

    GAS_matrix = modify.get_specific_matrix(GAS, len(GAS), n)
    print('GAS_matrix:', GAS_matrix)

    M = get_M(max(m, n))
    print('M:', M)

    # 判断是否需要进行调整
    subsets = modify.get_subsets(max(m, n))
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
    print('E1:', E1)
    # print('det(E):', np.linalg.det(E))
    # print('det(GF(E)):', np.linalg.det(GF(E)))

    E_inv = np.linalg.inv(GF(E))
    print('E_inv:', E_inv)

    rands = [GF(np.random.randint(1, 256)) for _ in range(max(max(m, n), len(M)))]  # 增加随机数干扰项，提高安全性
    print('rands:', rands)

    # 如果m < n或len(M) > n，要给s添加随机数，显式地添加
    if m < n or (m == n and len(M) > n):
        if m < n:
            k = n - m
        else:
            k = len(M) - n
        for _ in range(k):
            s.append([np.random.randint(256) for _ in range(len(s[0]))])
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
    # print('shares2:', shares2)
    to_save = np.array(shares2)
    for i in range(len(to_save)):
        Image.fromarray(to_save[i].reshape(shape).astype(np.uint8)) \
            .save("./result/mLTn/share_{}.bmp".format(i + 1))

    # 如果m>n或m<n进行了调整，要对GAS进行更新
    w = get_w(E1, M, GAS)

    s_recover = []
    for i in range(len(shares)):
        s_recover.append(recover(w, shares[i], GAS))
    # print('s_recover:', s_recover)

    s_recover_ultimate = get_recover_ultimate(s_recover, max(max(m, n), len(M)), rands)
    s_recover_ultimate = s_recover_ultimate[: m]  # m<n时，只需要取前m个秘密
    # print('s_recover_ultimate:', s_recover_ultimate)
    to_save2 = np.array(s_recover_ultimate)
    for i in range(len(to_save2)):
        Image.fromarray(to_save2[i].reshape(shape).astype(np.uint8)) \
            .save("./result/mLTn/reconstruct_{}.bmp".format(i + 1))


if __name__ == '__main__':
    main()