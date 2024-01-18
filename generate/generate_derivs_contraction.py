import sympy as sp

x = sp.symbols("x", real=True)
y = sp.symbols("y", real=True)
z = sp.symbols("z", real=True)
a = sp.symbols("a", real=True, positive=True)
r2 = x**2 + y**2 + z**2
exp = sp.exp(-a * r2)

##########################
# Cartesian
###############################
cartesian_f = [
    x * x * x * exp,
    x * x * y * exp,
    x * x * z * exp,
    x * y * y * exp,
    x * y * z * exp,
    x * z * z * exp,
    y * y * y * exp,
    y * y * z * exp,
    y * z * z * exp,
    z * z * z * exp,
]
cartesian_g = [
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
][::-1]
sp.init_printing(latex=True)
for i, formula in enumerate(cartesian_g):
    print(formula)
    deriv_x = sp.simplify(sp.diff(sp.simplify(formula), x))
    deriv_y = sp.simplify(sp.diff(sp.simplify(formula), y))
    deriv_z = sp.simplify(sp.diff(sp.simplify(formula), z))
    deriv_x = deriv_x.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    deriv_y = deriv_y.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    deriv_z = deriv_z.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    for t in ["r_A_x", "r_A_y", "r_A_z"]:
        for numb in ["2"]:
            replace = t + "*" + t
            # deriv_x = str(deriv_x).replace(t + "**" + numb + ".0", replace)
            deriv_x = str(deriv_x).replace(t + "**" + numb,        replace)
            deriv_x = deriv_x.replace("**1.0", "")
            deriv_x = deriv_x.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")


            # deriv_y = str(deriv_y).replace(t + "**" + numb + ".0", replace)
            deriv_y = str(deriv_y).replace(t + "**" + numb,        replace)
            deriv_y = deriv_y.replace("**1.0", "")
            deriv_y = deriv_y.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")

            # deriv_z = str(deriv_z).replace(t + "**" + numb + ".0", replace)
            deriv_z = str(deriv_z).replace(t + "**" + numb,        replace)
            deriv_z = deriv_z.replace("**1.0", "")
            deriv_z = deriv_z.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")

        for numb in ["3", "4", "5"]:
            replace = "pow(" + t + ", " + numb + ")"
            # deriv_x = str(deriv_x).replace(t + "**" + numb + ".0", replace)
            deriv_x = str(deriv_x).replace(t + "**" + numb, replace)
            # deriv_y = str(deriv_y).replace(t + "**" + numb + ".0", replace)
            deriv_y = str(deriv_y).replace(t + "**" + numb, replace)
            # deriv_z = str(deriv_z).replace(t + "**" + numb + ".0", replace)
            deriv_z = str(deriv_z).replace(t + "**" + numb, replace)
    print(deriv_x + f" *   // d {formula} / dx")
    print(deriv_y + f" *   // d {formula} / dy")
    print(deriv_z + f" *   // d {formula} / dz")
    print("\n")


##########################
# SOLID HARMONICS
###############################

assert 1 == 0


# Get the solid harmonics for g-type (l=4),   ordering is []
solid_harmonics_f = [
    (5 * z**2 - 3 * r2) * z * exp/ 2,  # m = 0,
    (5 * z**2 - r2) * x   * exp/ 2,  # m = 1,
    (5 * z**2 - r2) * y   * exp/ 2,   #m = -1,
    (x**2 - y**2) * z  * exp / 2 ,   # m = 2
    x * y * z  * exp,    # m = -2
    (x**2 - 3 * y**2) * x  * exp/ 2,  # m = 3
    (3 * x**2 - y**2) * y   * exp/ 2
]
solid_harmonics_g = [
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


for i, formula in enumerate(solid_harmonics_g):
    print("M ", m_g[i])
    deriv_x = sp.simplify(sp.diff(sp.simplify(formula), x))
    deriv_y = sp.simplify(sp.diff(sp.simplify(formula), y))
    deriv_z = sp.simplify(sp.diff(sp.simplify(formula), z))
    deriv_x = deriv_x.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    deriv_y = deriv_y.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    deriv_z = deriv_z.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
    for t in ["r_A_x", "r_A_y", "r_A_z"]:
        for numb in ["2"]:
            replace = t + "*" + t
            # deriv_x = str(deriv_x).replace(t + "**" + numb + ".0", replace)
            deriv_x = str(deriv_x).replace(t + "**" + numb,        replace)
            deriv_x = deriv_x.replace("**1.0", "")
            deriv_x = deriv_x.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")


            # deriv_y = str(deriv_y).replace(t + "**" + numb + ".0", replace)
            deriv_y = str(deriv_y).replace(t + "**" + numb,        replace)
            deriv_y = deriv_y.replace("**1.0", "")
            deriv_y = deriv_y.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")

            # deriv_z = str(deriv_z).replace(t + "**" + numb + ".0", replace)
            deriv_z = str(deriv_z).replace(t + "**" + numb,        replace)
            deriv_z = deriv_z.replace("**1.0", "")
            deriv_z = deriv_z.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")

        for numb in ["3", "4", "5"]:
            replace = "pow(" + t + ", " + numb + ")"
            # deriv_x = str(deriv_x).replace(t + "**" + numb + ".0", replace)
            deriv_x = str(deriv_x).replace(t + "**" + numb, replace)
            # deriv_y = str(deriv_y).replace(t + "**" + numb + ".0", replace)
            deriv_y = str(deriv_y).replace(t + "**" + numb, replace)
            # deriv_z = str(deriv_z).replace(t + "**" + numb + ".0", replace)
            deriv_z = str(deriv_z).replace(t + "**" + numb, replace)
    print(deriv_x + " * ")
    print(deriv_y + " * ")
    print(deriv_z + " * ")
    print("\n")
