from decimal import Decimal as D


def get_n1_input(x1: D, x2: D) -> D:
    w_a_str = '-1'
    w_a = D(w_a_str)

    w_b_str = '0'
    w_b = D(w_b_str)

    result = x1 * w_a + x2 * w_b

    msg = f"N1_input:\n" \
          f"{x1} * ({w_a}) + {x2} * ({w_b}) = {result}"
    print(msg)

    return result


def get_n1_output(n1_input: D) -> D:
    msg = f"N1_output = N1_input = {n1_input}"
    print(msg)

    return n1_input


def get_n2_input(x1: D, x2: D) -> D:
    w_a_str = '2'
    w_a = D(w_a_str)

    w_b_str = '1'
    w_b = D(w_b_str)

    result = x1 * w_a + x2 * w_b

    msg = f"N2_input:\n" \
          f"{x1} * ({w_a}) + {x2} * ({w_b}) = {result}"
    print(msg)

    return result


def get_n2_output(n2_input: D) -> D:
    msg = f"N2_output = N2_input = {n2_input}"
    print(msg)

    return n2_input


def get_n3_input(x1: D, x2: D) -> D:
    w_a_str = '-2'
    w_a = D(w_a_str)

    w_b_str = '1'
    w_b = D(w_b_str)

    result = x1 * w_a + x2 * w_b

    msg = f"N3_input:\n" \
          f"{x1} * ({w_a}) + {x2} * ({w_b}) = {result}"
    print(msg)

    return result


def get_n3_output(n3_input: D) -> D:
    msg = f"N3_output = N3_input = {n3_input}"
    print(msg)

    return n3_input


def get_n4_input(n1_output: D, n2_output: D, n3_output: D):
    w_a_str = '2'
    w_a = D(w_a_str)

    w_b_str = '-0.5'
    w_b = D(w_b_str)

    w_c_str = '1'
    w_c = D(w_c_str)

    result = n1_output * w_a + n2_output * w_b + n3_output * w_c

    msg = f"N4_input:\n" \
          f"{n1_output} * ({w_a}) + {n2_output} * {w_b} + {n2_output} * ({n3_output})= {result}"
    print(msg)

    return result


def get_n4_output(n4_input: D) -> D:
    msg = f"N4_output = N3_input = {n4_input}"
    print(msg)

    return n4_input


def get_n5_input(n1_output: D, n2_output: D, n3_output: D):
    w_a_str = '-2'
    w_a = D(w_a_str)

    w_b_str = '1'
    w_b = D(w_b_str)

    w_c_str = '0.5'
    w_c = D(w_c_str)

    result = n1_output * w_a + n2_output * w_b + n3_output * w_c

    msg = f"N5_input:\n" \
          f"{n1_output} * ({w_a}) + {n2_output} * {w_b} + {n2_output} * ({n3_output})= {result}"
    print(msg)

    return result


def get_n5_output(n5_input: D) -> D:
    msg = f"N5_output = N3_input = {n5_input}"
    print(msg)

    return n5_input


def network(x1, x2):
    # The network graph: https://i.imgur.com/cSKwz3D.png
    x1 = D(str(x1))
    x2 = D(str(x2))

    # n1
    n1_input = get_n1_input(x1, x2)
    n1_output = get_n1_output(n1_input)
    print()

    # n2
    n2_input = get_n2_input(x1, x2)
    n2_output = get_n2_output(n2_input)
    print()

    # n3
    n3_input = get_n3_input(x1, x2)
    n3_output = get_n3_output(n3_input)
    print()

    # n4
    n4_input = get_n4_input(n1_output, n2_output, n3_output)
    n4_output = get_n4_output(n4_input)
    print()

    # n5
    n5_input = get_n5_input(n1_output, n2_output, n3_output)
    n5_output = get_n5_output(n5_input)
    print()

    n4_output = float(n4_output)
    n5_output = float(n5_output)

    msg = f"Output = ({n4_output}, {n5_output})"
    print(msg)

    return n4_output, n5_output


if __name__ == '__main__':
    # network(0.5, 1)
    network(1, 2)
