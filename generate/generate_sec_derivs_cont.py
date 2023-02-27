import sympy as sp

x = sp.symbols("x", real=True)
y = sp.symbols("y", real=True)
z = sp.symbols("z", real=True)
a = sp.symbols("a", real=True, positive=True)
r2 = x**2 + y**2 + z**2
exp = sp.exp(-a * r2)

# Get the solid harmonics for g-type (l=4),   ordering is []
s_cartesian = [
    exp
]
p_cartesian = [
    x * exp, y * exp, z * exp
]
d_cartesian = [
    x * x * exp, y * y * exp, z * z * exp, x * y * exp, x * z * exp, y * z * exp
]
d_spherical = [
    (2 * z * z - x * x - y * y) * exp / 2,   #m = 0
    x * z * exp,  # m =1,
    y * z * exp,  # m = -1,
    (x * x - y * y) * exp / 2,  # m = 2
    x * y * exp,    # m = -2
]
f_cartesian = [
    x * x * x * exp,
    y * y * y * exp,
    z * z * z * exp,
    x * y * y * exp,
    x * x * y * exp,
    x * x * z * exp,
    x * z * z * exp,
    y * z * z * exp,
    y * y * z * exp,
    x * y * z * exp
]
f_spherical = [
    (5 * z**2 - 3 * r2) * z * exp/ 2,  # m = 0,
    (5 * z**2 - r2) * x   * exp/ 2,  # m = 1,
    (5 * z**2 - r2) * y   * exp/ 2,   #m = -1,
    (x**2 - y**2) * z  * exp / 2 ,   # m = 2
    x * y * z  * exp,    # m = -2
    (x**2 - 3 * y**2) * x  * exp/ 2,  # m = 3
    (3 * x**2 - y**2) * y   * exp/ 2
]
g_cartesian = [
    z * z * z * z * exp,
    y * z * z * z * exp,
    y * y * z * z * exp,
    y * y * y * z * exp,
    y * y * y * y * exp,
    x * z * z * z * exp,
    x * y * z * z * exp,
    x * y * y * z * exp,
    x * y * y * y * exp,
    x * x * z * z * exp,
    x * x * y * z * exp,
    x * x * y * y * exp,
    x * x * x * z * exp,
    x * x * x * y * exp,
    x * x * x * x * exp
]
g_spherical = [
    (35 * z**4 - 30 * z**2 * r2 + 3 * r2**2) * exp / 8,  #m= 0
    (7 * z**2 - 3 * r2) * x * z * exp / 2,  # m = 1
    (7 * z**2 - 3 * r2) * y * z * exp/ 2,  # m = -1
    (7 * z**2 - r2) * (x**2 - y**2) * exp / 4,  #m = 2
    (7 * z**2 - r2) * x * y * exp / 2,             # m = -2
    (x**2 - 3 * y**2) * x * z * exp / 2,  # m = 3
    (3 * x**2 - y**2) * y * z * exp / 2,  # m = -3
    (x**4 - 6 * x**2 * y**2 + y**4) * exp / 8,  # m = 4
    (x**2 - y**2) * x * y * exp / 2  # m = -4
]
m_g = [0, 1, -1, 2, -2, 3, -3, 4, -4]

total_ord = [(0, 0),
             (1, 0), (1, 1), (1, -1),
             (2, 0), (2, 1), (2, -1), (2, )]
total = s_cartesian + p_cartesian + d_cartesian + d_spherical + \
    f_cartesian + f_spherical + g_cartesian + g_spherical

for i, formula in enumerate(total):

    sec_deriv_x = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), x)), x))
    sec_deriv_y = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), y)), y))
    sec_deriv_z = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), z)), z))
    total = sp.simplify(sec_deriv_x + sec_deriv_y + sec_deriv_z)
    # sec_deriv_x = sec_deriv_x.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    # sec_deriv_y = sec_deriv_y.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    # sec_deriv_z = sec_deriv_z.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})

    total = total.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    #print(total)
    for t in ["r_A_x", "r_A_y", "r_A_z"]:
        for numb in ["2"]:
            replace = t + "*" + t
            # deriv_x = str(deriv_x).replace(t + "**" + numb + ".0", replace)
            total = str(total).replace(t + "**" + numb,        replace)
            total = total.replace("**1.0", "")
            total = total.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")
        for numb in ["3", "4", "5", "6"]:
            replace = "pow(" + t + ", " + numb + ")"
            total = str(total).replace(t + "**" + numb, replace)

    total = total.replace("alpha**2", "alpha*alpha")

    #print(str(sec_deriv_x) + " *  \n" + str(sec_deriv_y) + " *  \n" + str(sec_deriv_z))

    print(total + " * ")
    print("\n")
