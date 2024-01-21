r"""
This file generates C++ Code that computes the integrals of the point-charge
between two primitive Gaussians between all kinds of sub-shells, e.g. s-px, px-dxx, py-px.

Only handles d-orbitals.
"""

# The shells
# CHANGE HERE IF GONNA ADD. ORDER MATTERS ALL LOT.
subshells = ["s",
             "px", "py", "pz",
             "dxx", "dyy", "dzz", "dxy", "dxz", "dyz",
             'fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'
             ]
shells = ["s", "p", "d", "f"]
shells_numb = {"s" : 0, "p" : 1, "d" : 2, "f": 3}
# The angular momentum component
# CHANGE HERE IF GONNA ADD>
angmom_components = {"s" : (0, 0, 0),
                     "px" : (1, 0, 0), "py" : (0, 1, 0), "pz" : (0, 0, 1),
                     "dxx" : (2, 0, 0), "dxy" : (1, 1, 0), "dxz" : (1, 0, 1), "dyy" : (0, 2, 0),
                        "dyz" : (0, 1, 1), "dzz" : (0, 0, 2),
                     "fxxx": (3, 0, 0), "fxxy":  (2, 1, 0), "fxxz": (2, 0, 1), "fxyy": (1, 2, 0), "fxyz": (1, 1, 1), "fxzz": (1, 0, 2),
                        "fyyy": (0, 3, 0), "fyyz": (0, 2, 1), "fyzz": (0, 1, 2), "fzzz": (0, 0, 3)
                     }
# CHANGE HERE IF GONAN ADD, ORDER HERE MATTERS ALOT AND SHOULD MATCH EACH OTHER
angmom_components_list = {"s" : [(0, 0, 0)],
                          "p" : [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                          "d" : [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
                          "dp" : [(2, 0), (2, 1), (2, 1), (2, 2), (2, 2)],
                          "f" : [ (3, 0, 0), (0, 3, 0),  (0, 0, 3),  (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
                          "fp": [(3, 0), (3, 1), (3, 1), (3, 2), (3, 2), (3, 3), (3, 3)]}
# CHANGE HERE IF GONNA ADD. ORDER HERE MATTERS ALOT AND SHOULD MATCH SUBSHELLS.
subshells_str = {"s" : ["s"], "p" : ["px", "py", "pz"], "d" : ["dxx", "dyy", "dzz", "dxy", "dxz", "dyz"],
                 "f": ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']}
# Here the sphericals are split into the cartesian component ie [dxx, dyy, dzz, dxy, dxz, dyz].  Without normalization
# taken from iodata homepage
spherical_to_cartesian_d = {"c20" : [-0.5, -0.5, 1, 0, 0, 0.],
                          "c21" : [0, 0, 0, 0, 3**0.5, 0],
                          "s21" : [0, 0, 0, 0, 0, 3**0.5],
                          "c22" : [3**0.5 / 2., 0-3**0.5 / 2, 0, 0,  0, 0],
                          "s22" : [0, 0, 0, 3**0.5, 0, 0]}
spherical_list_d = ["c20", "c21", "s21", "c22", "s22"]
cartesian_list_d = ["dxx", "dyy", "dzz", "dxy", "dxz", "dyz"]

spherical_to_cartesian_f = {"c30" : [0.0, 0.0, 1.0, 0.0, 0.0, -3.0 / 2.0, 0.0, 0.0, -3.0 / 2.0, 0.0],
                            "c31" : [-6.0**0.5 / 4.0, 0.0, 0.0, -6.0**0.5 / 4.0, 0.0, 0.0, 6.0**0.5, 0.0, 0.0, 0.0],
                            "s31" : [0.0, -6.0**0.5 / 4.0, 0.0, 0.0, -6.0**0.5 / 4.0, 0.0, 0.0, 6.0**0.5, 0.0, 0.0],
                            "c32" : [0.0, 0.0, 0.0, 0.0, 0.0, 15.0**0.5 / 2.0, 0.0, 0.0, -15.0**0.5 / 2.0, 0.0],
                            "s32" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0**0.5],
                            "c33" : [10.0**0.5 / 4.0, 0.0, 0.0, -3.0 * 10.0**0.5 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "s33" : [0.0, -10.0**0.5 / 4.0, 0.0, 0.0, 3.0 * 10.0**0.5 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
spherical_list_f = ["c30", "c31", "s31", "c32", "s32", "c33", "s33"]
cartesian_list_f = ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']


function_header = lambda shell1 : "__device__ void compute_row_" + shell1 + \
                                  "_type_integral(const double3& A, const double3& pt,\n" \
                                  "    const int& numb_primitives1, double* d_point_charge, \n" \
                                  "    const int& point_index, int& i_integral, const int& iconst, int& jconst,\n" \
                                  "    const int& row_index, const int& npoints,\n" \
                                  "    const int& numb_contracted_shells, const int& icontr_shell, \n"\
                                  "    const double& screen_tol) {\n"
indent = lambda n : " " * n  # Function to compute the indentation.
function_initial = "" \
"   // Enumerate through second basis set starting right after the contracted shell. \n" \
"  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {\n" \
"    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};\n"\
"    int numb_primitives2 = (int) g_constant_basis[jconst++];\n"\
"    int angmom_2 = (int) g_constant_basis[jconst++];\n"\
"    // Enumerate through all primitives.\n"\
"    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {\n"\
"      double alpha = g_constant_basis[iconst + i_prim1];\n"\
"      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {\n"\
"        double beta = g_constant_basis[jconst + i_prim2];\n"\
"        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),\n"\
"                     (alpha * A.y + beta * B.y) / (alpha + beta),\n"\
"                     (alpha * A.z + beta * B.z) / (alpha + beta)};\n"

# CHANGE HERE IF GONNA ADD MORE
def write_normalization_line(ang_numb, exponent, angular_comp=None):
    # Exponent is either "alpha" or "beta"
    if (ang_numb == 0):
        return "gbasis::normalization_primitive_s(" + exponent +  ") *\n"
    elif (ang_numb == 1):
        return "gbasis::normalization_primitive_p("+ exponent + ") *\n"
    elif (ang_numb == 2):
        if angular_comp == (2, 0, 0):
            return "gbasis::normalization_primitive_d(" + exponent + ", 2, 0, 0) *\n"
        elif angular_comp == (0, 2, 0):
            return "gbasis::normalization_primitive_d(" + exponent + ", 0, 2, 0) *\n"
        elif angular_comp == (0, 0, 2):
            return "gbasis::normalization_primitive_d(" + exponent + ", 0, 0, 2) *\n"
        elif angular_comp == (1, 1, 0):
            return "gbasis::normalization_primitive_d(" + exponent + ", 1, 1, 0) *\n"
        elif angular_comp == (1, 0, 1):
            return "gbasis::normalization_primitive_d(" + exponent + ", 1, 0, 1) * \n"
        elif angular_comp == (0, 1, 1):
            return "gbasis::normalization_primitive_d(" + exponent + ", 0, 1, 1) *\n"
    elif (ang_numb == -2):
        return "gbasis::normalization_primitive_pure_d(" + exponent + ") * \n"
    elif (ang_numb == 3):
        if angular_comp == (3, 0, 0):
            return "gbasis::normalization_primitive_f(" + exponent + ", 3, 0, 0) *\n"
        elif angular_comp == (2, 1, 0):
            return "gbasis::normalization_primitive_f(" + exponent + ", 2, 1, 0) *\n"
        elif angular_comp == (2, 0, 1):
            return "gbasis::normalization_primitive_f(" + exponent + ", 2, 0, 1) *\n"
        elif angular_comp == (1, 2, 0):
            return "gbasis::normalization_primitive_f(" + exponent + ", 1, 2, 0) *\n"
        elif angular_comp == (1, 1, 1):
            return "gbasis::normalization_primitive_f(" + exponent + ", 1, 1, 1) *\n"
        elif angular_comp == (1, 0, 2):
            return "gbasis::normalization_primitive_f(" + exponent + ", 1, 0, 2) *\n"
        elif angular_comp == (0, 3, 0):
            return "gbasis::normalization_primitive_f(" + exponent + ", 0, 3, 0) *\n"
        elif angular_comp == (0, 2, 1):
            return "gbasis::normalization_primitive_f(" + exponent + ", 0, 2, 1) *\n"
        elif angular_comp == (0, 1, 2):
            return "gbasis::normalization_primitive_f(" + exponent + ", 0, 1, 2) *\n"
        elif angular_comp == (0, 0, 3):
            return "gbasis::normalization_primitive_f(" + exponent + ", 0, 0, 3) *\n"
    elif (ang_numb == -3):
        return "gbasis::normalization_primitive_pure_f(" + exponent + ") * \n"
    assert 1 == 0




file1 = open("integral_delete.cu", "w+")
header_files = []

def integral_rows_from_s_to_f_cartesian():
    def helper_integral_shell_to_spherical_harmonic_pure(sub_shell, type_ang):
        r"""
        Responsible for creating integrationg function from cartesian to pure d-orbitals.
        e.g. s-dxx, s-dyy, s-dzz, etc.  Note that it relies on transformation
        from Pure to Cartesian.

        If you're going to add f-orbitals, then you'll need to change these.
        """
        assert sub_shell in ["s", "px", "py", "pz", "dxx", "dxy", "dyy", "dxz", "dyz", "dzz",
                             'fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'
                             ]
        if type_ang == "d":
            angular_momentum = -2
            spherical_list_todo = spherical_list_d  # e.g. "c20", "c21", "s21", "c22", "s22"
            spherical_to_cartesian_todo = spherical_to_cartesian_d
            cartesian_list_todo = cartesian_list_d
            subshells_str_todo = subshells_str["d"]  # e.g. "dxx", "dyy", "dzz", "dxy", "dxz", "dyz"
        elif type_ang == "f":
            angular_momentum = -3
            spherical_list_todo = spherical_list_f  # e.g. "c30", "c31", "s31", "c32", "s32", "c33", "s33"
            spherical_to_cartesian_todo = spherical_to_cartesian_f
            cartesian_list_todo = cartesian_list_f   # 'fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', ...
            subshells_str_todo = subshells_str["f"]  # e.g. 'fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', ...

        # Write out the integrals.
        for i, spherical in enumerate(spherical_list_todo):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(i) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(angmom_a, "alpha", angmom_components_a))
            file1.write(indent(15) + write_normalization_line(angular_momentum, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for j, cartesian_expansion in enumerate(spherical_to_cartesian_todo[spherical]):
                # If it is non-zero, coefficient in the expansion of pure into cartesian write it out.
                if abs(cartesian_expansion) > 1e-8:
                    if is_first:
                        file1.write(indent(15))
                        is_first = False
                    else:
                        file1.write(indent(15) + " + ")

                    index_i, index_j = subshells.index(sub_shell), subshells.index(cartesian_list_todo[j])
                    if index_i <= index_j:
                        a, b = sub_shell, cartesian_list_todo[j]
                        file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                    else:
                        a, b = cartesian_list_todo[j], sub_shell
                        file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")

            file1.write(indent(15) + ");\n")

    for ia, subshell_a in enumerate(subshells):
        angmom_components_a = angmom_components[subshell_a] # Gets (i, j, k) Components
        angmom_a = sum(angmom_components_a)
    
        # Write out the function header and the initial boiler plate. Need to consider the diagonal.
        file1.write(function_header(subshell_a))
        # CHANGE HERE IF GONNA ADD MORE
        file1.write(function_initial)

        # Add integral screening derived via e^(-ab * (A - B)^2/(a + b)) < tol then integral is discarded.
        file1.write(indent(8) + "if (pow(A.x - B.x, 2.0) + pow(A.y - B.y, 2.0) + pow(A.z - B.z, 2.0) < -log(screen_tol) * "
                                "(alpha + beta) / (alpha * beta))  {\n")
        file1.write(indent(8) + "switch(angmom_2){\n")
        # Go through the next shells.
        for ib, shell_b in enumerate(shells):
            angmom_b = shells_numb[shell_b]  # Gets the integer 0, 1, 2, ... for the angular momentum
            # Go Through the different subshells possible of this shell.
            file1.write(indent(10) + "case " + str(angmom_b) + ": \n" )
            for i, angmom_components_b in enumerate(angmom_components_list[shell_b]):
                subshell_b = subshells_str[shell_b][i]
                file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+str(i)+") * npoints] +=\n")
                file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
                file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
                file1.write(indent(15) + write_normalization_line(angmom_a, "alpha", angmom_components_a))
                file1.write(indent(15) + write_normalization_line(angmom_b, "beta", angmom_components_b))
                # Swap the arguments because off-diagonal components are the same! and so only have px-py and not py-px.
                if (ia <= subshells.index(subshell_b)):
                    file1.write(indent(15) + "gbasis::compute_" + subshell_a + "_" + subshell_b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P);\n")
                else:
                    file1.write(indent(15) + "gbasis::compute_" + subshell_b + "_" + subshell_a + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P);\n")
            file1.write(indent(13) + "break;\n")
        # Spherical Harmonics.
        file1.write(indent(10) + "case -2: \n")
        helper_integral_shell_to_spherical_harmonic_pure(subshell_a, "d")
        file1.write(indent(13) + "break;\n")
        file1.write(indent(10) + "case -3: \n")
        helper_integral_shell_to_spherical_harmonic_pure(subshell_a, "f")
        file1.write(indent(13) + "break;\n")
        file1.write(indent(8) + "} // End switch\n ")
        file1.write(indent(8) + "} // End integral screening \n")

        file1.write(indent(6) + "}// End primitive 2\n")
        file1.write(indent(4) + "}// End primitive 1 \n")

        file1.write(indent(4) + "// Update index to go to the next segmented shell.\n")
        file1.write(indent(4) + "switch(angmom_2){\n")
        # CHANGE HERE IF GONNA ADD MORE
        file1.write(indent(6) + "case 0: i_integral += 1;\n")  # S-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 1: i_integral += 3;\n")  # P-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 2: i_integral += 6;\n")  # D-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 3: i_integral += 10;\n")  # F-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -2: i_integral += 5;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -3: i_integral += 7;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(4) + "} // End switch \n")
    
        file1.write(indent(2) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
        file1.write(indent(4) + "jconst += 2 * numb_primitives2;\n")
        file1.write(indent(2) + "}// End contracted shell 2\n")
        # Add closing brackets
        file1.write("}\n\n")



def integral_rows_of_pure_d(type_ang):
    if type_ang == "d":
        angular_momentum = -2
        spherical_list_todo = ["c20", "c21", "s21", "c22", "s22"]
        spherical_to_cartesian_todo = spherical_to_cartesian_d
        cartesian_list_todo = cartesian_list_d
    elif type_ang == "f":
        angular_momentum = -3
        spherical_list_todo = ["c30", "c31", "s31", "c32", "s32", "c33", "s33"]
        spherical_to_cartesian_todo = spherical_to_cartesian_f
        cartesian_list_todo = cartesian_list_f
    assert type_ang == "d"
    # Write out the integrals.
    for spherical1 in spherical_list_todo:
        # Write out function header.
        file1.write(function_header(spherical1))
        file1.write(function_initial)

        file1.write(indent(8) + "switch(angmom_2){\n")

        # CARTESIAN ONLY: Go through the shells ["s", "p", "d"]
        for ib, shell_b in enumerate( ["s", "p", "d", "f"]):
            angmom_b = shells_numb[shell_b]  # Gets the integer 0, 1, 2, ... for the angular momentum

            # Go Through the different subshells possible of this shell. This
            file1.write(indent(10) + "case " + str(angmom_b) + ": \n" )
            for i, angmom_components_b in enumerate(angmom_components_list[shell_b]):
                subshell_b = subshells_str[shell_b][i]
                angmom_b = sum(angmom_components_b)
                file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(i) +") * npoints] +=\n")
                file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
                file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
                file1.write(indent(15) + write_normalization_line(angular_momentum, "alpha"))
                file1.write(indent(15) + write_normalization_line(angmom_b, "beta", angmom_components_b))
                file1.write(indent(15) + "(\n")
                # Get The Cartesian expansion.
                is_first = True
                for j, cartesian_expansion in enumerate(spherical_to_cartesian_todo[spherical1]):
                    subshella = cartesian_list_todo[j]
                    # If it is non-zero write it out.
                    if abs(cartesian_expansion) > 1e-8:
                        # Get the correct spacing and addition
                        if is_first:
                            file1.write(indent(15))
                            is_first = False
                        else:
                            file1.write(indent(15) + " +" )

                        # The integral shouldn't swap when dealing with f-orbitals since f > d in ordering.
                        if (shell_b != "d" or i <= j) and shell_b != "f":
                            a, b = subshell_b, subshella
                            file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                        else:
                            # Swap integrals and Gaussian information
                            a, b = subshella, subshell_b
                            file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")


                file1.write(indent(15) + ");\n")
            file1.write(indent(13) + "break;\n")

        # Do the Pure D to Pure D case:
        file1.write(indent(10) + "case -2: \n" )
        for j, spherical2 in enumerate(spherical_list_d):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(-2, "alpha"))
            file1.write(indent(15) + write_normalization_line(-2, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian_d[spherical1]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian_d[spherical2]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            # Figure out which integral order to do.
                            if (k <= l):
                                a, b = cartesian_list_d[k], cartesian_list_d[l]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list_d[l], cartesian_list_d[k]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                        else:
                            file1.write(indent(15) +" + ")
                            if (k <= l):
                                a, b = cartesian_list_d[k], cartesian_list_d[l]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list_d[l], cartesian_list_d[k]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
            file1.write(indent(15) + ");\n")
        file1.write(indent(13) + "break;\n")

        # Go through Pure-D to Pure-F
        file1.write(indent(10) + "case -3: \n" )
        for j, spherical2 in enumerate(spherical_list_f):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(-2, "alpha"))
            file1.write(indent(15) + write_normalization_line(-3, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian_d[spherical1]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian_f[spherical2]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if not is_first:
                            file1.write(indent(15) +" + ")
                        else:
                            file1.write(indent(15))
                            is_first = False
                        # Swap integrals and Gaussian information.
                        a, b = cartesian_list_d[k], cartesian_list_f[l]
                        file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
            file1.write(indent(15) + ");\n")
        file1.write(indent(13) + "break;\n")


        file1.write(indent(8) + "} // End switch\n ")
        file1.write(indent(6) + "}// End primitive 2\n")
        file1.write(indent(4) + "}// End primitive 1 \n")


        file1.write(indent(4) + "// Update index to go to the next segmented shell.\n")
        file1.write(indent(4) + "switch(angmom_2){\n")
        # CHANGE HERE IF GONNA ADD MORE
        file1.write(indent(6) + "case 0: i_integral += 1;\n")  # S-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 1: i_integral += 3;\n")  # P-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 2: i_integral += 6;\n")  # D-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 3: i_integral += 10;\n")  # f-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -2: i_integral += 5;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -3: i_integral += 7;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(4) + "} // End switch \n")

        file1.write(indent(2) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
        file1.write(indent(4) + "jconst += 2 * numb_primitives2;\n")
        file1.write(indent(2) + "}// End contracted shell 2\n")
        # Add closing brackets
        file1.write("}\n\n")


def integral_rows_of_pure_f(type_ang):
    if type_ang == "d":
        angular_momentum = -2
        spherical_list_todo = ["c20", "c21", "s21", "c22", "s22"]
        spherical_to_cartesian_todo = spherical_to_cartesian_d
        cartesian_list_todo = cartesian_list_d
    elif type_ang == "f":
        angular_momentum = -3
        spherical_list_todo = ["c30", "c31", "s31", "c32", "s32", "c33", "s33"]
        spherical_to_cartesian_todo = spherical_to_cartesian_f
        cartesian_list_todo = cartesian_list_f
    assert type_ang == "f"
    # Write out the integrals.
    for spherical1 in spherical_list_todo:
        # Write out function header.
        file1.write(function_header(spherical1))
        file1.write(function_initial)

        file1.write(indent(8) + "switch(angmom_2){\n")

        # CARTESIAN ONLY: Go through the shells ["s", "p", "d"]
        for ib, shell_b in enumerate( ["s", "p", "d", "f"]):
            angmom_b = shells_numb[shell_b]  # Gets the integer 0, 1, 2, ... for the angular momentum

            # Go Through the different subshells possible of this shell. This
            file1.write(indent(10) + "case " + str(angmom_b) + ": \n" )
            for i, angmom_components_b in enumerate(angmom_components_list[shell_b]):
                subshell_b = subshells_str[shell_b][i]
                angmom_b = sum(angmom_components_b)
                file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(i) +") * npoints] +=\n")
                file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
                file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
                file1.write(indent(15) + write_normalization_line(angular_momentum, "alpha"))
                file1.write(indent(15) + write_normalization_line(angmom_b, "beta", angmom_components_b))
                file1.write(indent(15) + "(\n")
                # Get The Cartesian expansion.
                is_first = True
                for j, cartesian_expansion in enumerate(spherical_to_cartesian_todo[spherical1]):
                    subshella = cartesian_list_todo[j]
                    # If it is non-zero write it out.
                    if abs(cartesian_expansion) > 1e-8:
                        # Get the correct spacing and addition
                        if is_first:
                            file1.write(indent(15))
                            is_first = False
                        else:
                            file1.write(indent(15) + " +" )

                        # The integral shouldn't swap when dealing with f-orbitals since f > d in ordering.
                        index_i, index_j = subshells.index(subshell_b), subshells.index(cartesian_list_todo[j])
                        if index_i <= index_j: #shell_b in ["s", "p", "d"] or (shell_b == "f" or i > j):
                            # Swap them
                            a, b = subshell_b, subshella
                            file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                        else:
                            # Swap integrals and Gaussian information
                            a, b = subshella, subshell_b
                            file1.write(str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")


                file1.write(indent(15) + ");\n")
            file1.write(indent(13) + "break;\n")

        # Do the Pure F to Pure D case:
        file1.write(indent(10) + "case -2: \n" )
        for j, spherical2 in enumerate(spherical_list_d):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(-3, "alpha"))
            file1.write(indent(15) + write_normalization_line(-2, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian_f[spherical1]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian_d[spherical2]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            file1.write(indent(15))
                        else:
                            file1.write(indent(15) + " + ")
                        a, b = cartesian_list_d[l], cartesian_list_f[k]
                        file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
            file1.write(indent(15) + ");\n")
        file1.write(indent(13) + "break;\n")

        # Go through Pure-f to Pure-F
        file1.write(indent(10) + "case -3: \n" )
        for j, spherical2 in enumerate(spherical_list_f):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(-3, "alpha"))
            file1.write(indent(15) + write_normalization_line(-3, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian_f[spherical1]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian_f[spherical2]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            # Figure out which integral order to do.
                            if (k <= l):
                                a, b = cartesian_list_f[k], cartesian_list_f[l]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list_f[l], cartesian_list_f[k]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                        else:
                            file1.write(indent(15) +" + ")
                            if (k <= l):
                                a, b = cartesian_list_f[k], cartesian_list_f[l]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list_f[l], cartesian_list_f[k]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
            file1.write(indent(15) + ");\n")
        file1.write(indent(13) + "break;\n")
        file1.write(indent(8) + "} // End switch\n ")
        file1.write(indent(6) + "}// End primitive 2\n")
        file1.write(indent(4) + "}// End primitive 1 \n")
        file1.write(indent(4) + "// Update index to go to the next segmented shell.\n")
        file1.write(indent(4) + "switch(angmom_2){\n")
        # CHANGE HERE IF GONNA ADD MORE
        file1.write(indent(6) + "case 0: i_integral += 1;\n")  # S-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 1: i_integral += 3;\n")  # P-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 2: i_integral += 6;\n")  # D-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case 3: i_integral += 10;\n")  # f-type
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -2: i_integral += 5;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(6) + "case -3: i_integral += 7;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(4) + "} // End switch \n")

        file1.write(indent(2) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
        file1.write(indent(4) + "jconst += 2 * numb_primitives2;\n")
        file1.write(indent(2) + "}// End contracted shell 2\n")
        # Add closing brackets
        file1.write("}\n\n")
























# Do the diagonal of the integral array. This is precisely a shell integrated with itself.
function_diagonal_header = lambda shell1 : "__device__ void compute_diagonal_row_" + shell1 + \
                                           "_type_integral(const double3& A, const double3& pt,\n" \
                                            "    const int& numb_primitives1, double* d_point_charge, \n " \
                                            "    const int& point_index, int& i_integral, const int& iconst, int& jconst,\n " \
                                            "    const int& row_index, const int& npoints, const int& icontr_shell) {\n"
function_diagonal_initial = "" \
"  // Enumerate through all primitives.\n" \
"  for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {\n" \
"    double alpha = g_constant_basis[iconst + i_prim1];\n" \
"    for (int i_prim2 = 0; i_prim2 < numb_primitives1; i_prim2++) {\n" \
"      double beta = g_constant_basis[jconst + i_prim2];\n" \
"      double3 P = {(alpha * A.x + beta * A.x) / (alpha + beta),\n" \
"                   (alpha * A.y + beta * A.y) / (alpha + beta),\n" \
"                   (alpha * A.z + beta * A.z) / (alpha + beta)};\n"
# CHANGE HERE IF GONNA ADD MORE
subshells_left_str = {"py" : ["py", "pz"], "pz" : ["pz"],
                      "dyy" : ["dyy", "dzz", "dxy", "dxz", "dyz"],
                      "dzz" : ["dzz", "dxy", "dxz", "dyz"],
                      "dxy" : ["dxy", "dxz", "dyz"],
                      "dxz" : ["dxz", "dyz"],
                      "dyz" : ["dyz"],
                      'fyyy': ['fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fzzz': ['fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fxyy': ['fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fxxy': ['fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fxxz': ['fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fxzz': ['fxzz', 'fyzz', 'fyyz', 'fxyz'],
                      'fyzz': ['fyzz', 'fyyz', 'fxyz'],
                      'fyyz': ['fyyz', 'fxyz'],
                      'fxyz': ["fxyz"]
                      }
#
# CHANGE HERE IF GONNA ADD MORE ---->   "c20", "c21", "s21", "c22", "s22"
subshells_left = {"py" : [(0, 1, 0), (0, 0, 1)],
              "pz" : [(0, 0, 1)],
              "dxy" : [angmom_components[x] for x in subshells_left_str["dxy"]], #[(1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
              "dxz" : [angmom_components[x] for x in subshells_left_str["dxz"]],
              "dyy" : [angmom_components[x] for x in subshells_left_str["dyy"]],
              "dyz" : [angmom_components[x] for x in subshells_left_str["dyz"]],
              "dzz" : [angmom_components[x] for x in subshells_left_str["dzz"]],
              'fyyy': [angmom_components[x] for x in subshells_left_str["fyyy"]],
              'fzzz': [angmom_components[x] for x in subshells_left_str["fzzz"]],
              'fxyy': [angmom_components[x] for x in subshells_left_str["fxyy"]],
              'fxxy': [angmom_components[x] for x in subshells_left_str["fxxy"]],
              'fxxz': [angmom_components[x] for x in subshells_left_str["fxxz"]],
              'fxzz': [angmom_components[x] for x in subshells_left_str["fxzz"]],
              'fyzz': [angmom_components[x] for x in subshells_left_str["fyzz"]],
              'fyyz': [angmom_components[x] for x in subshells_left_str["fyyz"]],
              'fxyz': [angmom_components[x] for x in subshells_left_str["fxyz"]],
}
# Change here order matters here
for subshell_a in ["py", "pz", "dyy", "dzz", "dxy", "dxz", "dyz", 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']:
    angmom_components_a = angmom_components[subshell_a]
    angmom_a = sum(angmom_components_a)
    print(angmom_components_a)

    # Write out the function header and the initial boiler plate.
    file1.write(function_diagonal_header(subshell_a))
    file1.write(function_diagonal_initial)

    # Go through the following subshells .
    number_times = len(subshells_left_str[subshell_a])  # Number of integrals to do
    for ib, subshell_b in enumerate(subshells_left_str[subshell_a]):
        angmom_components_b = angmom_components[subshell_b]
        angmom_b = sum(angmom_components_b)
        file1.write(indent(6) + "//" + subshell_a + "-" + subshell_b + "\n")
        file1.write(indent(6) + "d_point_charge[point_index + (i_integral + "+str(ib)+") * npoints] +=\n")
        file1.write(indent(8) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
        file1.write(indent(8) + "g_constant_basis[jconst + numb_primitives1 + i_prim2] *\n")
        file1.write(indent(8) + write_normalization_line(angmom_a, "alpha", angmom_components_a))
        file1.write(indent(8) + write_normalization_line(angmom_b, "beta", angmom_components_b))
        file1.write(indent(8) + "gbasis::compute_" + subshell_a + "_" + subshell_b + "_nuclear_attraction_integral(alpha, A, beta, A, pt, P);\n")

    file1.write(indent(5) + "}// End primitive 2\n")
    file1.write(indent(3) + "}// End primitive 1 \n")
    file1.write(indent(3) + "// Update index to go to the next segmented shell.\n")
    file1.write(indent(3) + "i_integral += " + str(number_times) + ";\n")

    # The diagonal doesn't go through alls segmented shell and so it shouldnt update.
    # file1.write(indent(1) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
    # file1.write(indent(1) + "jconst += numb_primitives2 + numb_segment_shells2 + numb_segment_shells2 * numb_primitives2;\n")
    # Add closing brackets
    file1.write("}\n\n")


def write_row_spherical_harmonics_d(type_angmom):
    """Wirte out the smae thing but pure harmonics F/D-type only"""
    if type_angmom == "d":
        subshells_left = {"c21" : ["c21", "s21", "c22", "s22"],
                          "s21" : ["s21", "c22", "s22"],
                          "c22" : ["c22", "s22"],
                          "s22" : ["s22"]}
        angular_momentum = -2
        todo = ["c21", "s21", "c22", "s22"]
        spherical_to_cartesian_todo = spherical_to_cartesian_d
        cartesian_list_todo = cartesian_list_d
    elif type_angmom == "f":
        subshells_left = {"c31": ["c31", "s31", "c32", "s32", "c33", "s33"],
                          "s31": ["s31", "c32", "s32", "c33", "s33"],
                          "c32": ["c32", "s32", "c33", "s33"],
                          "s32": ["s32", "c33", "s33"],
                          "c33": ["c33", "s33"],
                          "s33": ["s33"]}
        angular_momentum = -3
        todo = ["c31", "s31", "c32", "s32", "c33", "s33"]
        spherical_to_cartesian_todo = spherical_to_cartesian_f
        cartesian_list_todo = cartesian_list_f
    # Here the sphericals are split into the cartesian component ie [dxx, dxy, dxz, dyy, dyz, dzz].
    for spherical_a in todo:
        file1.write(function_diagonal_header(spherical_a))
        file1.write(function_diagonal_initial)

        # Go through the following subshells .
        number_times = len(subshells_left[spherical_a])  # Number of integrals to do
        for j, spherical_b in enumerate(subshells_left[spherical_a]):
            file1.write(indent(6) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(8) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(8) + "g_constant_basis[jconst + numb_primitives1 + i_prim2] *\n")
            file1.write(indent(8) + write_normalization_line(angular_momentum, "alpha"))
            file1.write(indent(8) + write_normalization_line(angular_momentum, "beta"))
            file1.write(indent(8) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian_todo[spherical_a]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian_todo[spherical_b]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            # Figure out which integral order to do.
                            if (k <= l):
                                a, b = cartesian_list_todo[k], cartesian_list_todo[l]
                                file1.write(indent(8) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, A, pt, P)\n")
                            else:
                                a, b = cartesian_list_todo[l], cartesian_list_todo[k]
                                file1.write(indent(8) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, A, alpha, A, pt, P)\n")
                        else:
                            file1.write(indent(8) +" + ")
                            if (k <= l):
                                a, b = cartesian_list_todo[k], cartesian_list_todo[l]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, A, pt, P)\n")
                            else:
                                a, b = cartesian_list_todo[l], cartesian_list_todo[k]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, A, alpha, A, pt, P)\n")
            file1.write(indent(8) + ");\n")

        file1.write(indent(5) + "}// End primitive 2\n")
        file1.write(indent(3) + "}// End primitive 1 \n")
        file1.write(indent(3) + "// Update index to go to the next segmented shell.\n")
        file1.write(indent(3) + "i_integral += " + str(number_times) + ";\n")
        file1.write("}\n\n")



integral_rows_from_s_to_f_cartesian()
integral_rows_of_pure_d("d")
integral_rows_of_pure_f("f")
write_row_spherical_harmonics_d("d")
write_row_spherical_harmonics_d("f")
file1.close()
