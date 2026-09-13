#include "basis_function_array.hpp"
#include "orbitals_description.hpp"
#include "atomic_data.hpp"
#include "molecular_geometry.hpp"
#include "array_helpers.hpp"
#include "eigenvalues_eigenvectors.hpp"
#include "build_arrays.hpp"
#include "converge.hpp"
#include "compute_energies.hpp"
#include "orbital_shader_creator.hpp"

#include <iostream>
#include <string>

/*
 For implementing MP2, 
 I've followed the Wikipedia page "Moller-Plesset perturbation theory",
 as well as Chapter 23, pg 833 of Boudreau and Swanson (see README).
*/
/* double mp2(const array_helpers::Array1D &energies, 
           const array_helpers::Array2D &orbitals, 
           const array_helpers::HypercubeArray &repulsion_exchange,
           int occupation_count) {
    int total_count = orbitals.row_size();
    double sum = 0.0;
    for (int occ_ind1 = 0; occ_ind1 < occupation_count; occ_ind1++) {
        for (int occ_ind2 = 0; occ_ind2 < occupation_count; occ_ind2++) {
            for (int virt_ind1 = occupation_count; 
                 virt_ind1 < total_count; virt_ind1++) {
                for (int virt_ind2 = virt_ind1; 
                     virt_ind2 < total_count; virt_ind2++) {
                    double occ_e1 = energies(occ_ind1);
                    double occ_e2 = energies(occ_ind2);
                    double virt_e1 = energies(virt_ind1);
                    double virt_e2 = energies(virt_ind2);
                    double occ12_virt12 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(occ_ind1), orbitals.c_ptr(occ_ind2),
                        1, 3,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2));
                    // double virt12_occ12 = occ12_virt12;
                    double virt12_occ12 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2),
                        1, 3,
                        orbitals.c_ptr(occ_ind1), orbitals.c_ptr(occ_ind2));
                    double virt12_occ21 = repulsion_exchange.reduce(
                        0, 2,
                        orbitals.c_ptr(virt_ind1), orbitals.c_ptr(virt_ind2),
                        1, 3,
                        orbitals.c_ptr(occ_ind2), orbitals.c_ptr(occ_ind1));
                    double 
                    term = 2.0*occ12_virt12*virt12_occ12/(
                        occ_e1 + occ_e2 - virt_e1 - virt_e2);
                    term -= occ12_virt12*virt12_occ21/(
                        occ_e1 + occ_e2 - virt_e1 - virt_e2); 
                    // sum += term;
                    sum += ((virt_ind2 == virt_ind1)? term: 2.0*term);
                }
            }
        }
    }
    return sum;
}*/

/* array_helpers::Array2D construct_ci_matrix(
    const array_helpers::Array1D &energies,
    const array_helpers::Array2D &orbitals,
    const array_helpers::SquareArray &kinetic,
    const array_helpers::SquareArray &nuclear,
    const array_helpers::HypercubeArray &repulsion_exchange,
    int occupation_count
) {
    int total_count = orbitals.row_size();
    for (int i = occupation_count; i < total_count; i++) {

    }
    double kinetic_energy = get_kinetic_energy(kinetic, orbitals, orbitals);
    double nuclear_potential = get_nuclear_potential_energy(nuclear, orbitals, orbitals);
    get_repulsion_exchange_energy(repulsion_exchange, orbitals, orbitals);


}*/


void closed_shell_element_example(int z) {
    spatial::Vector position {.t=0.0, .x=0.0, .y=0.0, .z=0.0};
    NuclearChargesArray nuclear_charges {
        {{position, z}}};
    OrbitalsData descr;
    switch (z) {
        case 1:
        descr = atomic_data_descriptions::ORB_1P1E_1S21;
        case 2:
        descr = atomic_data_descriptions::ORB_2P2E_1S4;
        break;
        case 3:
        descr = atomic_data_descriptions::ORB_3P3E_1S4_2S4;
        break;
        case 4:
        descr = atomic_data_descriptions::ORB_4P4E_1S4_2S4;
        break;
        case 5:
        descr = atomic_data_descriptions::ORB_5P5E_1S4_2S4_2P4;
        break;
        case 6:
        descr = atomic_data_descriptions::ORB_6P6E_1S4_2S4_2P4;
        break;
        case 7:
        descr = atomic_data_descriptions::ORB_7P7E_1S4_2S4_2P4;
        break;
        case 8:
        descr = atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5;
        break;
        case 9:
        descr = atomic_data_descriptions::ORB_9P9E_1S4_2S4_2P4;
        break;
        case 10:
        descr = atomic_data_descriptions::ORB_10P10E_1S6_2S6_2P6;
        break;
        case 11:
        descr = atomic_data_descriptions::ORB_11P11E_1S4_2S4_2P4_3S4;
        break;
        case 12:
        descr = atomic_data_descriptions::ORB_12P12E_1S4_2S4_2P4_3S4;
        break;
        case 13:
        descr = atomic_data_descriptions::ORB_13P13E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 14:
        descr = atomic_data_descriptions::ORB_14P14E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 15:
        descr = atomic_data_descriptions::ORB_15P15E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 16:
        descr = atomic_data_descriptions::ORB_16P16E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 17:
        descr = atomic_data_descriptions::ORB_17P17E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 18:
        descr = atomic_data_descriptions::ORB_18P18E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 19:
        descr = atomic_data_descriptions::ORB_19P19E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 20:
        descr = atomic_data_descriptions::ORB_20P20E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 21:
        descr = atomic_data_descriptions::ORB_21P21E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 22:
        descr = 
            atomic_data_descriptions::ORB_22P22E_1S4_2S4_2P4_3S4_3P4_3D211_4S211;
        break;
        case 23:
        descr = 
            atomic_data_descriptions::ORB_23P23E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 24:
        descr = 
            atomic_data_descriptions::ORB_24P24E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 25:
        descr = 
            atomic_data_descriptions::ORB_25P25E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 26:
        descr = 
            atomic_data_descriptions::ORB_26P26E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 27:
        descr = 
            atomic_data_descriptions::ORB_27P27E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 28:
        descr = 
            atomic_data_descriptions::ORB_28P28E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 29:
        descr = 
            atomic_data_descriptions::ORB_29P29E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 30:
        descr = 
            atomic_data_descriptions::ORB_30P30E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 31:
        descr = 
            atomic_data_descriptions::ORB_31P31E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 32:
        descr = 
            atomic_data_descriptions::ORB_32P32E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 33:
        descr = 
            atomic_data_descriptions::ORB_33P33E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 34:
        descr = 
            atomic_data_descriptions::ORB_34P34E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 35:
        descr = 
            atomic_data_descriptions::ORB_35P35E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 36:
        descr = 
            atomic_data_descriptions::ORB_36P36E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
    }

    orbital_description_data::PositionedOrbitalsData 
    element {descr, position};
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {element});
    int n = arr.get_number_of_basis_functions();
    // printf("Number of basis functions: %d.\n", n);
    // if (z == 22)
    //     arr.print();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, 
        arr, nuclear_charges);
    int count = z/2;
    if (z == 1)
        count = 1;
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(count);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            count, {element});
    array_helpers::Array2D orbitals_final
        = array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    /* for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }*/
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        converge::iteration(energies, orbitals, overlap, h,
                repulsion_exchange, orbitals);
        if (i == 19)
            converge::iteration(energies_final, orbitals_final, overlap, h,
                repulsion_exchange, orbitals);
        // if (i == 19) {
        //     printf("Orbital energies:\n");
        //     for (int k = 0; k < energies.size(); k++)
        //         printf("%g \n", energies_final(k));
        // }
    }
    printf("Koopmans' ionization energy (eV): %g\n",
         -27.211386245*energies(energies.size() - 1));
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    // double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 1);
    // printf("kinetic and potential energies: %g, %g\n", ke, pe);
    printf("Total energy: %g\n", ke + pe + re + ne // + mp2e
    );
    /* for (int i = 0; i < orbitals_final.col_size(); i++) {
        for (int j = 0; j < orbitals_final.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }*/
}

void unrestricted_element_example(int z, bool verbose=false) {
spatial::Vector position {.t=0.0, .x=0.0, .y=0.0, .z=0.0};
    NuclearChargesArray nuclear_charges {
        {{position, z}}};
    OrbitalsData descr;
    int u_count, d_count;
    switch (z) {
        case 1:
        u_count = 1, d_count = 0;
        descr = atomic_data_descriptions::ORB_1P1E_1S21;
        case 2:
        u_count = 1, d_count = 1;
        descr = atomic_data_descriptions::ORB_2P2E_1S4;
        break;
        case 3:
        u_count = 2, d_count = 1;
        descr = atomic_data_descriptions::ORB_3P3E_1S4_2S4;
        break;
        case 4:
        u_count = 2, d_count = 2;
        descr = atomic_data_descriptions::ORB_4P4E_1S4_2S4;
        break;
        case 5:
        u_count = 3, d_count = 2;
        descr = atomic_data_descriptions::ORB_5P5E_1S4_2S4_2P4;
        break;
        case 6:
        u_count = 4, d_count = 2;
        descr = atomic_data_descriptions::ORB_6P6E_1S5_2S5_2P5;
        break;
        case 7:
        u_count = 5, d_count = 2;
        descr = atomic_data_descriptions::ORB_7P7E_1S4_2S4_2P4;
        break;
        case 8:
        u_count = 5, d_count = 3;
        descr = atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5;
        break;
        case 9:
        u_count = 5, d_count = 4;
        descr = atomic_data_descriptions::ORB_9P9E_1S4_2S4_2P4;
        break;
        case 10:
        u_count = 5, d_count = 5;
        descr = atomic_data_descriptions::ORB_10P10E_1S6_2S6_2P6;
        break;
        case 11:
        u_count = 6, d_count = 5;
        descr = atomic_data_descriptions::ORB_11P11E_1S4_2S4_2P4_3S4;
        break;
        case 12:
        u_count = 6, d_count = 6;
        descr = atomic_data_descriptions::ORB_12P12E_1S4_2S4_2P4_3S4;
        break;
        case 13:
        u_count = 7, d_count = 6;
        descr = atomic_data_descriptions::ORB_13P13E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 14:
        u_count = 8, d_count = 6;
        descr = atomic_data_descriptions::ORB_14P14E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 15:
        u_count = 9, d_count = 6;
        descr = atomic_data_descriptions::ORB_15P15E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 16:
        u_count = 9, d_count = 7;
        descr = atomic_data_descriptions::ORB_16P16E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 17:
        u_count = 9, d_count = 8;
        descr = atomic_data_descriptions::ORB_17P17E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 18:
        u_count = 9, d_count = 9;
        descr = atomic_data_descriptions::ORB_18P18E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 19:
        u_count = 10, d_count = 9;
        descr = atomic_data_descriptions::ORB_19P19E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 20:
        u_count = 10, d_count = 10;
        descr = atomic_data_descriptions::ORB_20P20E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 21:
        u_count = 11, d_count = 10;
        descr = atomic_data_descriptions::ORB_21P21E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 22:
        u_count = 12, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_22P22E_1S4_2S4_2P4_3S4_3P4_3D211_4S211;
        break;
        case 23:
        u_count = 13, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_23P23E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 24:
        u_count = 14, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_24P24E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 25:
        u_count = 15, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_25P25E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 26:
        u_count = 15, d_count = 11;
        descr = 
            atomic_data_descriptions::ORB_26P26E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 27:
        u_count = 15, d_count = 12;
        descr = 
            atomic_data_descriptions::ORB_27P27E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 28:
        u_count = 15, d_count = 13;
        descr = 
            atomic_data_descriptions::ORB_28P28E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 29:
        u_count = 15, d_count = 14;
        descr = 
            atomic_data_descriptions::ORB_29P29E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 30:
        u_count = 15, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_30P30E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 31:
        u_count = 16, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_31P31E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 32:
        u_count = 17, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_32P32E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 33:
        u_count = 18, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_33P33E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 34:
        u_count = 18, d_count = 16;
        descr = 
            atomic_data_descriptions::ORB_34P34E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 35:
        u_count = 18, d_count = 17;
        descr = 
            atomic_data_descriptions::ORB_35P35E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 36:
        u_count = 18, d_count = 18;
        descr = 
            atomic_data_descriptions::ORB_36P36E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
    }
    orbital_description_data::PositionedOrbitalsData 
    element {descr, position};
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {element});
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);

    struct timespec frame_time[2];
    clock_gettime(CLOCK_MONOTONIC, &frame_time[0]);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, 
        arr, nuclear_charges);
    clock_gettime(CLOCK_MONOTONIC, &frame_time[1]);
    double delta_t = frame_time[1].tv_sec - frame_time[0].tv_sec;
    if (n > 25)
        std::cout << "Construction time: " << delta_t << "s \n";

    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies_up(u_count);
    array_helpers::Array2D orbitals_up
        = orbital_description_data::get_orbital_basis_function_coefficients(
            u_count, {element});
    array_helpers::Array1D energies_down(d_count);
    array_helpers::Array2D orbitals_down
        = orbital_description_data::get_orbital_basis_function_coefficients(
            d_count, {element});
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    converge::open(
        energies_up, orbitals_up, u_count,
        energies_down, orbitals_down, d_count,
        overlap, h, repulsion_exchange, 20, verbose);
    // for (int i = 0; i < 20; i++) {
    //     if (z == 1) {
    //         iteration(
    //             energies_up, overlap, h,
    //             energies_up, orbitals_up);
    //     } else {
    //         iteration(
    //             energies_up, orbitals_up, 
    //             energies_down, orbitals_down,
    //             overlap, 
    //             h, repulsion_exchange,
    //             orbitals_up, orbitals_down);
    //     }
    //     if (i == 19) {
    //         printf("Orbital energies:\n");
    //         for (int k = 0; k < energies_up.size(); k++) {
    //             printf("%g \n", energies_up(k));
    //             if (d_count > 0 && k < energies_down.size())
    //                 printf("%g \n", energies_down(k));
    //         }
    //     }
    // }
    double up_last = energies_up(energies_up.size() - 1);
    double down_last = energies_down(energies_down.size() - 1);
    printf("Koopmans' ionization energy (eV): %g\n",
         -27.211386245*std::max(up_last, down_last));
    array_helpers::Array2D orbitals = array_helpers::row_stack(
        orbitals_up, orbitals_down);
    // printf("Orbitals up dimensions: %d, %d,\n", 
    //     orbitals_up.row_count(), orbitals_up.column_count());
    // printf("Orbitals down dimensions: %d, %d,\n", 
    //     orbitals_down.row_count(), orbitals_down.column_count());
    // printf("Orbitals dimensions: %d, %d,\n", 
    //     orbitals.row_count(), orbitals.column_count());
    // double ke_u = get_kinetic_energy(kinetic, orbitals_up);
    // double ke_d = get_kinetic_energy(kinetic, orbitals_down);
    // double pe_u =  get_nuclear_potential_energy(nuclear, orbitals_up);
    // double pe_d =  get_nuclear_potential_energy(nuclear, orbitals_down);
    double ke = compute_energies::kinetic(kinetic, orbitals)/2.0;
    double pe = compute_energies::nuclear_potential(nuclear, orbitals)/2.0;
    double re = compute_energies::repulsion(repulsion_exchange, orbitals);
    double ex_up = compute_energies::exchange(
        repulsion_exchange, orbitals_up);
    double ex_down = compute_energies::exchange(
        repulsion_exchange, orbitals_down);
    // printf("Kinetic Energy: %g\n", ke);
    // printf("Nuclear Energy: %g\n", pe);
    if (z == 1) {
        printf("Total energy: %g\n", ke + pe);
    } else {
        printf("Total energy: %g\n", ke + pe + (re - ex_up - ex_down)/2.0);
    }
    // for (int i = 0; i < orbitals.col_size(); i++) {
    //     for (int j = 0; j < orbitals.row_size(); j++) {
    //         printf("%g ", orbitals(i, j));
    //     }
    //     printf("\n");
    // }
}

void h2_example() {
    orbital_description_data::PositionedOrbitalsData h1
        {atomic_data_descriptions::ORB_1P1E_1S31,
         // atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
            {.t=0.0, 1.37,  0.0,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        {atomic_data_descriptions::ORB_1P1E_1S31, 
         // atomic_data_descriptions::ORB_1P1E_1S22_2S22_2P22,
            {.t=0.0, 0.0, 0.0,  0.0}}; 
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 1.37,  0.0,  0.0}, 1},
        {{.t=0.0, 0.0, 0.0,  0.0}, 1},
        // {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {h1, h2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    build_arrays::fill(
        overlap, kinetic, nuclear,
        repulsion_exchange, arr, nuclear_charges);
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(1);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            1, {h1, h2});
    array_helpers::Array2D orbitals_final
        = array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    converge::closed(
        energies, orbitals, 1, 
        overlap, h, repulsion_exchange, 20);
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    // double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 1);
    printf("Total energy: %g\n", ke + pe + re + ne // + mp2e
    );

}

void h2o_example() {
    orbital_description_data::PositionedOrbitalsData h1
        { // atomic_data_descriptions::ORB_1P1E_1S22,
        atomic_data_descriptions::ORB_1P1E_1S31,
        // atomic_data_descriptions::ORB_1P1E_1S4_2S4_2P4,
            {.t=0.0, -1.93044664,  0.82666546,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData h2
        {  // atomic_data_descriptions::ORB_1P1E_1S22,
         atomic_data_descriptions::ORB_1P1E_1S31,
        // atomic_data_descriptions::ORB_1P1E_1S4_2S4_2P4,
            {.t=0.0, 0.82666546, -1.93044664,  0.0}}; 
    orbital_description_data::PositionedOrbitalsData o 
        {// atomic_data_descriptions::ORB_10P10E_1S5_2S311_2P311,
         // atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         // atomic_data_descriptions::ORB_8P8E_1S4_2S4_2P4,
         atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5,
         // atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, -1.93044664,  0.82666546,  0.0}, 1},
        {{.t=0.0, 0.82666546, -1.93044664,  0.0}, 1},
        {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {h1, h2, o});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, arr, nuclear_charges);
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(5);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
            5, {h1, h2, o});
    array_helpers::Array2D orbitals_final = 
        array_helpers::Array2D(orbitals.row_size(), orbitals.row_size());
    array_helpers::Array1D energies_final(orbitals.row_size());
    // for (int i = 0; i < orbitals.col_size(); i++) {
    //     for (int j = 0; j < orbitals.row_size(); j++) {
    //         printf("%g ", orbitals(i, j));
    //     }
    //     printf("\n");
    // }
    // for (int i = 5; i < overlap.row_size(); i++)
    //     orbitals(4, i) = 1.0;
    for (int i = 0; i < 20; i++) {
        converge::iteration(energies, orbitals, overlap, h, 
            repulsion_exchange, orbitals);
        if (i == 19)
            converge::iteration(energies_final, orbitals_final, overlap, h,
                repulsion_exchange, orbitals);
        // for (int i = 0; i < energies.size(); i++)
        //     printf("%g ", energies(i));
        // printf("\n");
    }
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    // double mp2e = mp2(energies_final, orbitals_final, repulsion_exchange, 5);
    printf("Total energy: %g\n", ke + pe + re + ne // + mp2e
    );
    // for (int i = 0; i < orbitals.col_size(); i++) {
    //     for (int j = 0; j < orbitals.row_size(); j++) {
    //         printf("%g ", orbitals(i, j));
    //     }
    //     printf("\n");
    // }
}

void benzene_example() {
    int n_electrons = 42;
    std::vector<orbital_description_data::PositionedOrbitalsData> hydrogens;
    std::vector<orbital_description_data::PositionedOrbitalsData> carbons;
    NuclearChargesArray nuclear_charges {{}};
    double pi = 3.141592653589793;
    for (int i = 0; i < 6; i++) {
        double angle = pi/3.0;
        double r_c = 2.6;
        double r_h = 4.6;
        spatial::Vector c_pos {
            .t=0.0, .x=r_c*cos(i*angle), .y=r_c*sin(i*angle), .z=0.0};
        spatial::Vector h_pos {
            .t=0.0, .x=r_h*cos(i*angle), .y=r_h*sin(i*angle), .z=0.0};
        carbons.push_back(
            orbital_description_data::PositionedOrbitalsData
            {
                // atomic_data_descriptions::ORB_6P6E_1S5_2S5_2P5,
                // atomic_data_descriptions::ORB_6P6E_1S4_2S4_2P4,
                atomic_data_descriptions::ORB_6P6E_1S5_2S5_2P5,
                c_pos
            }
        );
        hydrogens.push_back(
            orbital_description_data::PositionedOrbitalsData
            {
                atomic_data_descriptions::ORB_1P1E_1S31,
                // atomic_data_descriptions::ORB_1P1E_1S21,
                h_pos
            }
        );
        nuclear_charges.push_back({h_pos, 1});
        nuclear_charges.push_back({c_pos, 6});
    }
    std::vector<orbital_description_data::PositionedOrbitalsData> atoms;
    for (const PositionedOrbitalsData &c: carbons)
        atoms.push_back(c);
    for (const PositionedOrbitalsData &h: hydrogens)
        atoms.push_back(h);
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        atoms);
    int n = arr.get_number_of_basis_functions();
    std::cout << "Number of basis functions: " << n << std::endl;
    // arr.print();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    // array_helpers::HypercubeArray repulsion_exchange(n);
    array_helpers::Symmetric4 repulsion_exchange(n);

    struct timespec frame_time[2];
    clock_gettime(CLOCK_MONOTONIC, &frame_time[0]);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, 
        arr, nuclear_charges);
    clock_gettime(CLOCK_MONOTONIC, &frame_time[1]);
    double delta_t = frame_time[1].tv_sec - frame_time[0].tv_sec;
    std::cout << "Construction time: " << delta_t << "s \n";

    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(n_electrons/2);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
        n_electrons/2, atoms);
    converge::closed(
        energies, orbitals, n_electrons/2, overlap,
        h, repulsion_exchange, 10);
    write_orbital_to_file(orbitals, arr);
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(
        nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    printf("Nuclear energy: %g\n", ne);
    printf("Total energy: %g\n", ke + pe + re + ne);
}

void co2_example() {
    orbital_description_data::PositionedOrbitalsData o1
        {// atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         // atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
         atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5,
            {.ind{0.0, -2.2, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData o2
        {// atomic_data_descriptions::ORB_8P8E_1S5_2S32_2P32,
         // atomic_data_descriptions::ORB_8P8E_1S6_2S6_2P3111,
         atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5,
            {.ind{0.0, 2.2, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData c
        {// atomic_data_descriptions::ORB_7P7E_1S6_2S42_2P42,
         atomic_data_descriptions::ORB_6P6E_1S5_2S5_2P2111,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, -2.2, 0.0,  0.0}, 8},
        {{.t=0.0, 2.2, 0.0,  0.0}, 8},
        {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}, 6},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {o1, c, o2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::Symmetric4 repulsion_exchange(n);
    printf("Basis function count: %d\n", n);
    printf("Total number of elements: %d\n", repulsion_exchange.get_size());

    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange,
        arr, nuclear_charges);
    // Total energy: -186.876
    // 6s
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("Overlap %d, %d: %g\n", i, j, overlap(i, j));
    //         printf("Kinetic %d, %d: %g\n", i, j, kinetic(i, j));
    //         printf("Nuclear %d, %d: %g\n", i, j, nuclear(i, j));
    //     }
    // }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(11);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
        11, {o1, c, o2});
    converge::closed(
        energies, orbitals, 11, overlap,
        h, repulsion_exchange, 20, true);
    // for (int i = 0; i < 20; i++) {
    //     iteration(energies, orbitals, overlap, h, 
    //         repulsion_exchange, orbitals);
    //     for (int i = 0; i < energies.size(); i++)
    //         printf("%g ", energies(i));
    //     printf("\n");
    // }
    write_orbital_to_file(orbitals, arr);
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    printf("Total energy: %g\n", ke + pe + re + ne);
    // for (int i = 0; i < orbitals.col_size(); i++) {
    //     for (int j = 0; j < orbitals.row_size(); j++) {
    //         printf("%g ", orbitals(i, j));

    //     }
    //     printf("\n");
    // }
}

void o2_example() {
    orbital_description_data::PositionedOrbitalsData o1
        {atomic_data_descriptions::ORB_8P8E_1S4_2S4_2P4,
            {.ind{0.0, 0.0, 0.0, 0.0}}};
    orbital_description_data::PositionedOrbitalsData o2
        {atomic_data_descriptions::ORB_8P8E_1S4_2S4_2P4,
            {.ind{0.0, 2.31, 0.0, 0.0}}};
    NuclearChargesArray nuclear_charges = NuclearChargesArray({
        {{.t=0.0, 0.0, 0.0,  0.0}, 8},
        {{.t=0.0, 2.31, 0.0,  0.0}, 8},
    });
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        {o1, o2});
    arr.print();
    int n = arr.get_number_of_basis_functions();
    array_helpers::SquareArray overlap(n);
    array_helpers::SquareArray kinetic(n);
    array_helpers::SquareArray nuclear(n);
    array_helpers::HypercubeArray repulsion_exchange(n);
    build_arrays::fill(
        overlap, kinetic, nuclear, repulsion_exchange, 
        arr, nuclear_charges);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("Overlap %d, %d: %g\n", i, j, overlap(i, j));
            printf("Kinetic %d, %d: %g\n", i, j, kinetic(i, j));
            printf("Nuclear %d, %d: %g\n", i, j, nuclear(i, j));
        }
    }
    array_helpers::SquareArray h = kinetic + nuclear;
    array_helpers::Array1D energies(8);
    array_helpers::Array2D orbitals
        = orbital_description_data::get_orbital_basis_function_coefficients(
        8, {o1, o2});
    // array_helpers::Array2D orbitals(11, 61);
    // orbitals(0, 0) = 1.0;
    // for (int i = 11; i < overlap.row_size(); i++)
    //     orbitals(10, i) = 0.0;

    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));
        }
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        converge::iteration(energies, orbitals, overlap, h, 
            repulsion_exchange, orbitals);
        for (int i = 0; i < energies.size(); i++)
            printf("%g ", energies(i));
        printf("\n");
    }
    double ke = compute_energies::kinetic(kinetic, orbitals);
    double pe = compute_energies::nuclear_potential(nuclear, orbitals);
    double re = compute_energies::repulsion_exchange(
        repulsion_exchange, orbitals);
    double ne = nuclear_charges.get_energy();
    printf("Total energy: %g\n", ke + pe + re + ne);
    for (int i = 0; i < orbitals.col_size(); i++) {
        for (int j = 0; j < orbitals.row_size(); j++) {
            printf("%g ", orbitals(i, j));

        }
        printf("\n");
    }
}

