from modify import *


def main():
    m = 4
    n = 4
    subsets = get_subsets(n)
    # for _ in range(n):
    #     subsets.pop(0)
    # print('subsets:', subsets)

    # u_matrix = generate_matrix(4, subsets)
    u_matrix = get_specific_matrix(subsets, len(subsets), m)
    # u_matrix = get_specific_matrix(subsets, len(subsets), n)
    print('u_matrix:', u_matrix)

    GAS = [[1, 2], [2, 3], [3, 4], [1, 2, 3, 4]]
    # GAS = [[1, 4], [1, 2, 4], [1, 2, 3], [1, 3]]
    # GAS = [[1, 4], [1, 2, 4], [1, 2, 3], [1, 3], [2, 4]]
    # GAS_matrix = get_GAS_matrix(GAS)
    GAS_matrix = get_specific_matrix(GAS, len(GAS), n)
    # print(np.linalg.det(GF(GAS_matrix)))
    # print([1, 1, 0, 0] in GAS_matrix)
    print('GAS_matrix:', GAS_matrix)

    v_matrix = get_v_matrix(u_matrix)
    print('v:', v_matrix)

    C = modify1(GAS_matrix, u_matrix, v_matrix)
    # C = modify2(GAS_matrix, u_matrix, m - n)
    # C = modify3(GAS_matrix, u_matrix, n - m)
    print('C:', C)

    A = get_A(C)
    print(A)
    print(np.linalg.det(GF(A)))


if __name__ == '__main__':
    main()
