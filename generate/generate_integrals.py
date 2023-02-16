r"""
This file generates C++ Code that computes the integrals of the point-charge
between two primitive Gaussians between all kinds of sub-shells, e.g. s-px, px-dxx, py-px.

Only handles d-orbitals.
"""

# The shells
# CHANGE HERE IF GONNA ADD. ORDER MATTERS ALL LOT.
subshells = ["s", "px", "py", "pz", "dxx", "dyy", "dzz", "dxy", "dxz", "dyz",]
shells = ["s", "p", "d"]
shells_numb = {"s" : 0, "p" : 1, "d" : 2}
# The angular momentum component
# CHANGE HERE IF GONNA ADD>
angmom_components = {"s" : (0, 0, 0), "px" : (1, 0, 0), "py" : (0, 1, 0), "pz" : (0, 0, 1),
                     "dxx" : (2, 0, 0), "dxy" : (1, 1, 0), "dxz" : (1, 0, 1), "dyy" : (0, 2, 0),
                     "dyz" : (0, 1, 1), "dzz" : (0, 0, 2)}
# CHANGE HERE IF GONAN ADD, ORDER HERE MATTERS ALOT AND SHOULD MATCH EACH OTHER
angmom_components_list = {"s" : [(0, 0, 0)], "p" : [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                          "d" : [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
                          "dp" : [(2, 0), (2, 1), (2, 1), (2, 2), (2, 2)]}
# CHANGE HERE IF GONNA ADD. ORDER HERE MATTERS ALOT AND SHOULD MATCH SUBSHELLS.
subshells_str = {"s" : ["s"], "p" : ["px", "py", "pz"], "d" : ["dxx", "dyy", "dzz", "dxy", "dxz", "dyz"]} 
# Here the sphericals are split into the cartesian component ie [dxx, dyy, dzz, dxy, dxz, dyz].
spherical_to_cartesian = {"c20" : [-0.5, -0.5, 1, 0, 0, 0.],
                          "c21" : [0, 0, 0, 0, 3**0.5, 0],
                          "s21" : [0, 0, 0, 0, 0, 3**0.5],
                          "c22" : [3**0.5 / 2., 0-3**0.5 / 2, 0, 0,  0, 0],
                          "s22" : [0, 0, 0, 3**0.5, 0, 0]}
spherical_list = ["c20", "c21", "s21", "c22", "s22"]
cartesian_list = ["dxx", "dyy", "dzz", "dxy", "dxz", "dyz"]



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
    assert 1 == 0




file1 = open("integral_delete.cu", "w+")
header_files = []

def integral_rows_from_s_to_d_cartesian():
    def helper_integral_shell_to_spherical_harmonic_pure_d(sub_shell):
        r"""
        Responsible for creating integrationg function from cartesian to pure d-orbitals.
        e.g. s-dxx, s-dyy, s-dzz, etc.  Note that it relies on transformation
        from Pure to Cartesian.

        If you're going to add f-orbitals, then you'll need to change these.
        """
        assert sub_shell in ["s", "px", "py", "pz", "dxx", "dxy", "dyy", "dxz", "dyz", "dzz"]
        angular_momentum = -2
        # Write out the integrals.
        for i, spherical in enumerate(spherical_list):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(i) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(angmom_a, "alpha", angmom_components_a))
            file1.write(indent(15) + write_normalization_line(angular_momentum, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for j, cartesian_expansion in enumerate(spherical_to_cartesian[spherical]):
                # If it is non-zero, coefficient in the expansion of pure into cartesian write it out.
                if abs(cartesian_expansion) > 1e-8:
                    if sub_shell not in subshells_str["d"] or subshells_str["d"].index(sub_shell) <= j:
                        a, b = sub_shell, cartesian_list[j]
                        if is_first:
                            file1.write(indent(15) + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            is_first = False
                        else:
                            file1.write(indent(15) +" + " + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                    else:
                        # Swap the integrals and the gaussian information
                        a, b = cartesian_list[j], sub_shell
                        if is_first:
                            file1.write(indent(15) + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                            is_first = False
                        else:
                            file1.write(indent(15) +" + " + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                    
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
        helper_integral_shell_to_spherical_harmonic_pure_d(subshell_a)
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
        file1.write(indent(6) + "case -2: i_integral += 5;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(4) + "} // End switch \n")
    
        file1.write(indent(2) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
        file1.write(indent(4) + "jconst += 2 * numb_primitives2;\n")
        file1.write(indent(2) + "}// End contracted shell 2\n")
        # Add closing brackets
        file1.write("}\n\n")



def integral_rows_of_d_pure():
    angular_momentum = -2
    # Write out the integrals.
    for spherical1 in ["c20", "c21", "s21", "c22", "s22"]:
        # Write out function header.
        file1.write(function_header(spherical1))
        file1.write(function_initial)

        file1.write(indent(8) + "switch(angmom_2){\n")

        # Go through the shells ["s", "p", "d"].
        for ib, shell_b in enumerate( ["s", "p", "d"]):
            angmom_b = shells_numb[shell_b]  # Gets the integer 0, 1, 2, ... for the angular momentum

            # Go Through the different subshells possible of this shell.
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
                for j, cartesian_expansion in enumerate(spherical_to_cartesian[spherical1]):
                    subshella =  cartesian_list[j]
                    # If it is non-zero write it out.
                    if abs(cartesian_expansion) > 1e-8:
                        if is_first:
                            if shell_b != "d" or i <= j:
                                a, b = subshell_b, subshella
                                file1.write(indent(15) + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information
                                a, b = subshella, subshell_b
                                file1.write(indent(15) + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            is_first = False
                        else:
                            if shell_b != "d" or i <= j:
                                a, b = subshell_b, subshella
                                file1.write(indent(15) +" + " + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information
                                a, b = subshella, subshell_b
                                file1.write(indent(15) +" + " + str(cartesian_expansion) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")


                file1.write(indent(15) + ");\n")
            file1.write(indent(13) + "break;\n")

        # Do the Pure D to Pure D case:
        file1.write(indent(10) + "case -2: \n" )
        for j, spherical2 in enumerate(spherical_list):
            file1.write(indent(13) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(15) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(15) + "g_constant_basis[jconst + numb_primitives2 + i_prim2] *\n")
            file1.write(indent(15) + write_normalization_line(-2, "alpha"))
            file1.write(indent(15) + write_normalization_line(-2, "beta"))
            file1.write(indent(15) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian[spherical1]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian[spherical2]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            # Figure out which integral order to do.
                            if (k <= l):
                                a, b = cartesian_list[k], cartesian_list[l]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list[l], cartesian_list[k]
                                file1.write(indent(15) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, B, alpha, A, pt, P)\n")
                        else:
                            file1.write(indent(15) +" + ")
                            if (k <= l):
                                a, b = cartesian_list[k], cartesian_list[l]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, B, pt, P)\n")
                            else:
                                # Swap integrals and Gaussian information.
                                a, b = cartesian_list[l], cartesian_list[k]
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
        file1.write(indent(6) + "case -2: i_integral += 5;\n")
        file1.write(indent(8) + "break;\n")
        file1.write(indent(4) + "} // End switch \n")

        file1.write(indent(2) + "// Update index of constant memory to the next contracted shell of second basis set. \n")
        file1.write(indent(4) + "jconst += 2 * numb_primitives2;\n")
        file1.write(indent(2) + "}// End contracted shell 2\n")
        # Add closing brackets
        file1.write("}\n\n")



integral_rows_from_s_to_d_cartesian()
integral_rows_of_d_pure()
























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
                      "dyz" : ["dyz"]}
#
# CHANGE HERE IF GONNA ADD MORE ---->   "c20", "c21", "s21", "c22", "s22"
subshells_left = {"py" : [(0, 1, 0), (0, 0, 1)],
              "pz" : [(0, 0, 1)],
              "dxy" : [angmom_components[x] for x in subshells_left_str["dxy"]], #[(1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
              "dxz" : [angmom_components[x] for x in subshells_left_str["dxz"]],
              "dyy" : [angmom_components[x] for x in subshells_left_str["dyy"]],
              "dyz" : [angmom_components[x] for x in subshells_left_str["dyz"]],
              "dzz" : [angmom_components[x] for x in subshells_left_str["dzz"]]
              }
# Change here order matters here
for subshell_a in ["py", "pz", "dyy", "dzz", "dxy", "dxz", "dyz"]:
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


def write_row_spherical_harmonics_d():
    """Wirte out the smae thing but pure harmonics D-type only"""
    subshells_left = {"c21" : ["c21", "s21", "c22", "s22"],
                      "s21" : ["s21", "c22", "s22"],
                      "c22" : ["c22", "s22"],
                      "s22" : ["s22"]}
    angular_momentum = -2
    # Here the sphericals are split into the cartesian component ie [dxx, dxy, dxz, dyy, dyz, dzz].
    for spherical_a in ["c21", "s21", "c22", "s22"]:
        file1.write(function_diagonal_header(spherical_a))
        file1.write(function_diagonal_initial)

        # Go through the following subshells .
        number_times = len(subshells_left[spherical_a])  # Number of integrals to do
        for j, spherical_b in enumerate(subshells_left[spherical_a]):
            file1.write(indent(6) + "d_point_charge[point_index + (i_integral + "+ str(j) +") * npoints] +=\n")
            file1.write(indent(8) + "g_constant_basis[iconst + numb_primitives1 + i_prim1] *\n")
            file1.write(indent(8) + "g_constant_basis[jconst + numb_primitives1 + i_prim2] *\n")
            file1.write(indent(8) + write_normalization_line(-2, "alpha"))
            file1.write(indent(8) + write_normalization_line(-2, "beta"))
            file1.write(indent(8) + "(\n")
            # Get The Cartesian expansion.
            is_first = True
            for k, cartesian_expansion in enumerate(spherical_to_cartesian[spherical_a]):
                for l, cartesian_expansion2 in enumerate(spherical_to_cartesian[spherical_b]):
                    # Check that both coefficients are non-zero.
                    if abs(cartesian_expansion) > 1e-8 and abs(cartesian_expansion2) > 1e-8:
                        if is_first:
                            is_first = False
                            # Figure out which integral order to do.
                            if (k <= l):
                                a, b = cartesian_list[k], cartesian_list[l]
                                file1.write(indent(8) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, A, pt, P)\n")
                            else:
                                a, b = cartesian_list[l], cartesian_list[k]
                                file1.write(indent(8) + str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, A, alpha, A, pt, P)\n")
                        else:
                            file1.write(indent(8) +" + ")
                            if (k <= l):
                                a, b = cartesian_list[k], cartesian_list[l]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(alpha, A, beta, A, pt, P)\n")
                            else:
                                a, b = cartesian_list[l], cartesian_list[k]
                                file1.write(str(cartesian_expansion) +  "*" +  str(cartesian_expansion2) + " * gbasis::compute_" + a + "_" + b + "_nuclear_attraction_integral(beta, A, alpha, A, pt, P)\n")
            file1.write(indent(8) + ");\n")

        file1.write(indent(5) + "}// End primitive 2\n")
        file1.write(indent(3) + "}// End primitive 1 \n")
        file1.write(indent(3) + "// Update index to go to the next segmented shell.\n")
        file1.write(indent(3) + "i_integral += " + str(number_times) + ";\n")
        file1.write("}\n\n")


write_row_spherical_harmonics_d()
file1.close()
