import sympy as sp
def write_normalization_line(ang_numb, exponent, angular_comp=None):
    # Exponent is either "alpha" or "beta"
    if (ang_numb == 0):
        return "chemtools::normalization_primitive_s(" + exponent +  ") *"
    elif (ang_numb == 1):
        return "chemtools::normalization_primitive_p("+ exponent + ") *"
    elif (ang_numb == 2):
        return "chemtools::normalization_primitive_d(" + exponent + f", {angular_comp[0]}, {angular_comp[1]}, {angular_comp[2]}) * "
    elif (ang_numb == -2):
        return "chemtools::normalization_primitive_pure_d(" + exponent + ") *"
    elif (ang_numb == 3):
        return "chemtools::normalization_primitive_f(" + exponent + f", {angular_comp[0]}, {angular_comp[1]}, {angular_comp[2]}) *"
    elif (ang_numb == -3):
        return "chemtools::normalization_primitive_pure_f(" + exponent + ") *"
    elif (ang_numb == -4):
        return "chemtools::normalization_primitive_pure_g(" + exponent + ") *"
    elif (ang_numb == 4):
        return "chemtools::normalization_primitive_g(" + exponent + f", {angular_comp[0]}, {angular_comp[1]}, {angular_comp[2]}) *"
    assert 1 == 0



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
    x * x * exp,
    y * y * exp,
    z * z * exp,
    x * y * exp,
    x * z * exp,
    y * z * exp
]
d_spherical = [
    (2 * z * z - x * x - y * y) * exp / 2,   #m = 0
    sp.sqrt(3.0) * x * z * exp,  # m =1,
    sp.sqrt(3.0) * y * z * exp,  # m = -1,
    sp.sqrt(3.0) * (x * x - y * y) * exp / 2,  # m = 2
    sp.sqrt(3.0) * x * y * exp,    # m = -2
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
    sp.sqrt(1.5) * (5 * z**2 - r2) * x   * exp/ 2,  # m = 1,
    sp.sqrt(1.5) * (5 * z**2 - r2) * y   * exp/ 2,   #m = -1,
    sp.sqrt(15.0) * (x**2 - y**2) * z  * exp / 2 ,   # m = 2
    sp.sqrt(15.0) * x * y * z  * exp,    # m = -2
    sp.sqrt(2.5) * (x**2 - 3 * y**2) * x  * exp/ 2,  # m = 3
    sp.sqrt(2.5) * (3 * x**2 - y**2) * y   * exp/ 2
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
    sp.sqrt(2.5) * (7 * z**2 - 3 * r2) * x * z * exp / 2,  # m = 1
    sp.sqrt(2.5) * (7 * z**2 - 3 * r2) * y * z * exp/ 2,  # m = -1
    sp.sqrt(5.0) * (7 * z**2 - r2) * (x**2 - y**2) * exp / 4,  #m = 2
    sp.sqrt(5.0) * (7 * z**2 - r2) * x * y * exp / 2,             # m = -2
    sp.sqrt(17.5) * (x**2 - 3 * y**2) * x * z * exp / 2,  # m = 3
    sp.sqrt(17.5) * (3 * x**2 - y**2) * y * z * exp / 2,  # m = -3
    sp.sqrt(35.0) *(x**4 - 6 * x**2 * y**2 + y**4) * exp / 8,  # m = 4
    sp.sqrt(35.0) *(x**2 - y**2) * x * y * exp / 2  # m = -4
]
m_g = [0, 1, -1, 2, -2, 3, -3, 4, -4]

total_ord = [(0, 0),
             (1, 0), (1, 1), (1, -1),
             (2, 0), (2, 1), (2, -1), (2, )]
total = s_cartesian + p_cartesian + d_cartesian + d_spherical + \
    f_cartesian + f_spherical + g_cartesian + g_spherical
subshells = ["s",
             "px", "py", "pz",
             "dxx", "dyy", "dzz", "dxy", "dxz", "dyz",
             "c20", "c21", "s21", "c22", "s22",
             'fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz',
             "c30", "c31", "s31", "c32", "s32", "c33", "s33",
             "gzzz", "gyzzz", "gyyzz", "gyyyz", "gyyyy", "gxzzz", "gxyzz", "gxyyz", "gxyyy", "gxxzz",
                "gxxyz", "gxxyy", "gxxxz", "gxxxy", "gxxxx",
             "c40", "c41", "s41", "c42", "s42", "c43", "s43", "c44", "s44"
             ]
angmom = {0: ["s"],
          1: ["px", "py", "pz",],
          2: [ "dxx", "dyy", "dzz", "dxy", "dxz", "dyz"],
          -2: ["c20", "c21", "s21", "c22", "s22"],
          3: ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz',],
          -3: ["c30", "c31", "s31", "c32", "s32", "c33", "s33"],
          4: [ "gzzz", "gyzzz", "gyyzz", "gyyyz", "gyyyy", "gxzzz", "gxyzz", "gxyyz", "gxyyy", "gxxzz",
               "gxxyz", "gxxyy", "gxxxz", "gxxxy", "gxxxx"],
          -4: [  "c40", "c41", "s41", "c42", "s42", "c43", "s43", "c44", "s44"]}
all_formulas = {
   0: s_cartesian, 1: p_cartesian, 2: d_cartesian, -2: d_spherical, 3: f_cartesian,
  -3: f_spherical, 4: g_cartesian, -4: g_spherical
}
angmom_components_list = {0 : [(0, 0, 0)],
                          1 : [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                          2 : [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
                          -2 : [(2, 0), (2, 1), (2, 1), (2, 2), (2, 2)],
                          3 : [ (3, 0, 0), (0, 3, 0),  (0, 0, 3),  (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
                          -3: [(3, 0), (3, 1), (3, 1), (3, 2), (3, 2), (3, 3), (3, 3)],
                          4 : [
                                (0, 0, 4), (0, 1, 3),  (0, 2, 2),  (0, 3, 1), (0, 4, 0), (1, 0, 3), (1, 1, 2), (1, 2, 1),
                              (1, 3, 0), (2, 0, 2), (2, 1, 1), (2, 2, 0), (3, 0, 1), (3, 1, 0), (0, 0, 4)],
                          -4: [(4, 0), (4, 1), (4, 1), (4, 2), (4, 2), (4, 3), (4, 3), (4, 4), (4, 4)]}
count = 0
for angmom, formula_list in angmom.items():
    print(f"if (angmom == {angmom}) {str('{')}")
    for i, formula in enumerate(all_formulas[angmom]):
        print(f"    // Taking sec derv (xx, xy, xz, yy, yz, zz) of {formula}")
        sec_deriv_xx = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), x)), x))
        sec_deriv_xy = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), x)), y))
        sec_deriv_xz = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), x)), z))
        sec_deriv_yy = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), y)), y))
        sec_deriv_yz = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), y)), z))
        sec_deriv_zz = sp.simplify(sp.diff(sp.simplify(sp.diff(sp.simplify(formula), z)), z))

        # sec_deriv_x = sec_deriv_x.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
        # sec_deriv_y = sec_deriv_y.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
        # sec_deriv_z = sec_deriv_z.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
        all_derivs = [sec_deriv_xx, sec_deriv_xy, sec_deriv_xz, sec_deriv_yy, sec_deriv_yz, sec_deriv_zz]
        stuff = ["xx", "xy", "xz", "yy", "yz", "zz"]
        for j, total in enumerate(all_derivs):

            print(f"    d_sec_deriv_contracs[knumb_points * (knumb_contractions*{j} + icontractions + {i}) + global_index] +=")
            print(f"        {write_normalization_line(angmom, 'alpha', angmom_components_list[angmom][i])} ")
            print(f"        coeff_prim * ")
            total = total.subs({x: "r_A_x", y: "r_A_y", z: "r_A_z", a: "alpha"})
            #print(total)
            for t in ["r_A_x", "r_A_y", "r_A_z", "(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)"]:
                for numb in ["2"]:
                    replace = t + "*" + t
                    total = str(total).replace(t + "**" + numb,        replace)
                    total = total.replace("**1.0", "")
                    total = total.replace("*exp(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z))", "")
                for numb in ["3"]:
                    replace = t + "*" + t + "*" + t
                    total = str(total).replace(t + "**" + numb,        replace)

                for numb in ["4", "5", "6"]:
                    replace = "pow(" + t + ", " + numb + ")"
                    total = str(total).replace(t + "**" + numb, replace)

            total = total.replace("alpha**2", "alpha*alpha")
            total = total.replace("sqrt(3)", "sqrt(3.0)")
            print(f"        {total}  * ")
            print(f"        exponential;")
    print(f"{str('}')}")
