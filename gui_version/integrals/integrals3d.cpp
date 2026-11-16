/* Compute the various integral relations between 3D Gaussian functions.
This is indebted to the following article:

    Joshua Goings, 
    A (hopefully) gentle guide to the computer implementation 
    of molecular integrals. 2017.
    https://joshuagoings.com/2017/04/28/integrals/

*/
#include "gaussian1d.hpp"
#include "integrals1d.hpp"
#include "gaussian3d.hpp"
#include "boys_function.hpp"
#include <cmath>
#include <iostream>

using namespace spatial;

#define PI 3.141592653589793
#define fp_t float

static void fill_overlap_coefficients(
    struct spatial::Vector *overlaps, 
    int index, const Gaussian1D &g1, const Gaussian1D &g2
) {

    fp_t r = g1.position() - g2.position();
    float e1 = g1.orbital_exponent();
    float e2 = g2.orbital_exponent();
    int a1 = g1.angular();
    int a2 = g2.angular();
    fp_t r2 = r*r;
    fp_t r3 = r2*r;
    fp_t r4 = r2*r2;
    fp_t r5 = r4*r;
    fp_t r6 = r3*r3;
    fp_t r7 = r6*r;
    fp_t r8 = r7*r;
    fp_t expVal = exp(-e1*e2*r2/(e1 + e2));

    fp_t e1_2 = e1*e1;
    fp_t e1_3 = e1_2*e1;
    fp_t e1_4 = e1_2*e1_2;
    fp_t e1_5 = e1_4*e1;
    fp_t e1_6 = e1_3*e1_3;
    fp_t e1_7 = e1_6*e1;
    fp_t e1_8 = e1_4*e1_4;

    fp_t e2_2 = e2*e2;
    fp_t e2_3 = e2_2*e2;
    fp_t e2_4 = e2_2*e2_2;
    fp_t e2_5 = e2_4*e2;
    fp_t e2_6 = e2_3*e2_3;
    fp_t e2_7 = e2_6*e2;
    fp_t e2_8 = e2_4*e2_4;

    fp_t e1e2_2 = (e1 + e2)*(e1 + e2);
    fp_t e1e2_3 = e1e2_2*(e1 + e2);
    fp_t e1e2_4 = e1e2_2*e1e2_2;
    fp_t e1e2_5 = e1e2_4*(e1 + e2);
    fp_t e1e2_6 = e1e2_3*e1e2_3;
    fp_t e1e2_7 = e1e2_6*(e1 + e2);
    fp_t e1e2_8 = e1e2_4*e1e2_4;

    int hexA1A2 = 16*a1 + a2;
    for (int i = 0; i < 8; i++)
        overlaps[i].ind[index] = 0.0;
    if (hexA1A2 == 0x00) {
        overlaps[0].ind[index] = expVal;
    }
    if (hexA1A2 == 0x01) {
        overlaps[0].ind[index] = e1*expVal*r/(e1 + e2);
        overlaps[1].ind[index] = 0.5*expVal/(e1 + e2);
    }
    if (hexA1A2 == 0x02) {
        overlaps[0].ind[index] = expVal*(0.5*e1 + e1_2*r2 + 0.5*e2)/e1e2_2;
        overlaps[1].ind[index] = 1.0*e1*expVal*r/e1e2_2;
        overlaps[2].ind[index] = 0.25*expVal/e1e2_2;
    }
    if (hexA1A2 == 0x03) {
        overlaps[0].ind[index] = e1*expVal*r*(1.5*e1 + e1_2*r2 + 1.5*e2)/e1e2_3;
        overlaps[1].ind[index] = expVal*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)/e1e2_3;
        overlaps[2].ind[index] = 0.75*e1*expVal*r/e1e2_3;
        overlaps[3].ind[index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x04) {
        overlaps[0].ind[index] = expVal*(1.0*e1_2*r2*(e1 + e2) + e1_2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))/e1e2_4;
        overlaps[1].ind[index] = e1*expVal*r*(3.0*e1 + 2.0*e1_2*r2 + 3.0*e2)/e1e2_4;
        overlaps[2].ind[index] = expVal*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2)/e1e2_4;
        overlaps[3].ind[index] = 0.5*e1*expVal*r/e1e2_4;
        overlaps[4].ind[index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x10) {
        overlaps[0].ind[index] = -e2*expVal*r/(e1 + e2);
        overlaps[1].ind[index] = 0.5*expVal/(e1 + e2);
    }
    if (hexA1A2 == 0x11) {
        overlaps[0].ind[index] = expVal*(-e1*e2*r2 + 0.5*e1 + 0.5*e2)/e1e2_2;
        overlaps[1].ind[index] = 0.5*expVal*r*(e1 - e2)/e1e2_2;
        overlaps[2].ind[index] = 0.25*expVal/e1e2_2;
    }
    if (hexA1A2 == 0x12) {
        overlaps[0].ind[index] = expVal*r*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2))/e1e2_3;
        overlaps[1].ind[index] = expVal*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2)/e1e2_3;
        overlaps[2].ind[index] = expVal*r*(0.5*e1 - 0.25*e2)/e1e2_3;
        overlaps[3].ind[index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x13) {
        overlaps[0].ind[index] = expVal*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))/e1e2_4;
        overlaps[1].ind[index] = expVal*r*(1.5*e1*(e1 + e2) + 0.5*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))/e1e2_4;
        overlaps[2].ind[index] = 0.75*expVal*(-e1*e2*r2 + e1 + e1_2*r2 + e2)/e1e2_4;
        overlaps[3].ind[index] = expVal*r*(0.375*e1 - 0.125*e2)/e1e2_4;
        overlaps[4].ind[index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x14) {
        overlaps[0].ind[index] = expVal*r*(1.5*e1*e2_2 + 5.25*e1_2*e2 - 3.0*e1_2*e2_2*r2 - 1.0*e1_3*e2*r2 + 3.0*e1_3 - 1.0*e1_4*e2*r4 + 2.0*e1_4*r2 - 0.75*e2_3)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[1].ind[index] = expVal*(3.75*e1*e2 - 3.0*e1*e2_2*r2 + 1.5*e1_2*e2*r2 + 1.875*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.5*e1_4*r4 + 1.875*e2_2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[2].ind[index] = expVal*r*(1.5*e1*(e1 + e2) + e1*(1.5*e1 + 1.0*e1_2*r2 + 1.5*e2) - e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2))/e1e2_5;
        overlaps[3].ind[index] = expVal*(-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2)/e1e2_5;
        overlaps[4].ind[index] = expVal*r*(0.25*e1 - 0.0625*e2)/e1e2_5;
        overlaps[5].ind[index] = 0.03125*expVal/e1e2_5;
    }
    if (hexA1A2 == 0x20) {
        overlaps[0].ind[index] = expVal*(0.5*e1 + 0.5*e2 + e2_2*r2)/e1e2_2;
        overlaps[1].ind[index] = -1.0*e2*expVal*r/e1e2_2;
        overlaps[2].ind[index] = 0.25*expVal/e1e2_2;
    }
    if (hexA1A2 == 0x21) {
        overlaps[0].ind[index] = expVal*r*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2))/e1e2_3;
        overlaps[1].ind[index] = expVal*(-0.5*e1*e2*r2 + 0.75*e1 - 0.5*e2*r2*(e1 - e2) + 0.75*e2)/e1e2_3;
        overlaps[2].ind[index] = expVal*r*(0.25*e1 - 0.5*e2)/e1e2_3;
        overlaps[3].ind[index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x22) {
        overlaps[0].ind[index] = expVal*(-1.0*e1*e2*r2*(e1 + e2) + 0.5*e1e2_2 - e2*r2*(1.0*e1*(e1 + e2) - e2*(0.5*e1 + e1_2*r2 + 0.5*e2)) + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2))/e1e2_4;
        overlaps[1].ind[index] = expVal*r*(0.5*e1*(e1 + e2) - 0.5*e2*(0.5*e1 + e1_2*r2 + 0.5*e2) - e2*(-1.0*e1*e2*r2 + 0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) + (e1 - 0.5*e2)*(e1 + e2))/e1e2_4;
        overlaps[2].ind[index] = expVal*(-0.5*e1*e2*r2 + 0.75*e1 + 0.25*e1_2*r2 - e2*r2*(0.5*e1 - 0.25*e2) + 0.75*e2)/e1e2_4;
        overlaps[3].ind[index] = 0.25*expVal*r*(e1 - e2)/e1e2_4;
        overlaps[4].ind[index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x23) {
        overlaps[0].ind[index] = expVal*r*(1.5*e1*e1e2_2 + 0.5*e1*(e1 + e2)*(1.5*e1 + e1_2*r2 + 1.5*e2) - e2*(e1 + e2)*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-e1*e2*r2*(1.5*e1 + e1_2*r2 + 1.5*e2) + 1.0*e1_2*r2*(e1 + e2) + 0.5*e1e2_2 + (e1 + e2)*(0.25*e1 + 0.5*e1_2*r2 + 0.25*e2)))/e1e2_5;
        overlaps[1].ind[index] = expVal*(3.75*e1*e2 - 3.75*e1*e2_2*r2 - 2.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 - 1.0*e1_3*e2*r4 + 2.25*e1_3*r2 + 1.875*e2_2 + 0.75*e2_3*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[2].ind[index] = expVal*r*(0.75*e1*(e1 + e2) + 0.25*e1*(1.5*e1 + e1_2*r2 + 1.5*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - 0.75*e2*(-e1*e2*r2 + e1 + e1_2*r2 + e2) + (e1 + e2)*(1.125*e1 - 0.375*e2))/e1e2_5;
        overlaps[3].ind[index] = expVal*(-0.375*e1*e2*r2 + 0.625*e1 + 0.375*e1_2*r2 - e2*r2*(0.375*e1 - 0.125*e2) + 0.625*e2)/e1e2_5;
        overlaps[4].ind[index] = expVal*r*(0.1875*e1 - 0.125*e2)/e1e2_5;
        overlaps[5].ind[index] = 0.03125*expVal/e1e2_5;
    }
    if (hexA1A2 == 0x24) {
        overlaps[0].ind[index] = expVal*(5.625*e1*e2_2 - 4.5*e1*e2_3*r2 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 + 3.0*e1_3*e2*r2 - 1.0*e1_3*e2_2*r4 + 1.875*e1_3 - 3.5*e1_4*e2*r4 + 1.0*e1_4*e2_2*r6 + 4.5*e1_4*r2 + 0.5*e1_5*r4 + 1.875*e2_3 + 0.75*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[1].ind[index] = expVal*r*(3.0*e1*e2_3*r2 + 11.25*e1_2*e2 - 6.0*e1_2*e2_2*r2 - 6.0*e1_3*e2*r2 + 2.0*e1_3*e2_2*r4 + 7.5*e1_3 - 1.0*e1_4*e2*r4 + 3.0*e1_4*r2 - 3.75*e2_3)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[2].ind[index] = expVal*(5.625*e1*e2 - 5.25*e1*e2_2*r2 - 1.5*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 - 2.0*e1_3*e2*r4 + 4.5*e1_3*r2 + 0.25*e1_4*r4 + 2.8125*e2_2 + 0.75*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[3].ind[index] = expVal*r*(0.75*e1*(e1 + e2) + e1*(0.75*e1 + 0.5*e1_2*r2 + 0.75*e2) - 0.5*e2*(0.75*e1 + 1.5*e1_2*r2 + 0.75*e2) - e2*(-0.5*e1*e2*r2 + 0.625*e1 + 0.75*e1_2*r2 + 0.625*e2) + (e1 - 0.25*e2)*(e1 + e2))/e1e2_6;
        overlaps[4].ind[index] = expVal*(-0.25*e1*e2*r2 + 0.46875*e1 + 0.375*e1_2*r2 - e2*r2*(0.25*e1 - 0.0625*e2) + 0.46875*e2)/e1e2_6;
        overlaps[5].ind[index] = expVal*r*(0.125*e1 - 0.0625*e2)/e1e2_6;
        overlaps[6].ind[index] = 0.015625*expVal/e1e2_6;
    }
    if (hexA1A2 == 0x30) {
        overlaps[0].ind[index] = e2*expVal*r*(-1.5*e1 - 1.5*e2 - e2_2*r2)/e1e2_3;
        overlaps[1].ind[index] = expVal*(0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)/e1e2_3;
        overlaps[2].ind[index] = -0.75*e2*expVal*r/e1e2_3;
        overlaps[3].ind[index] = 0.125*expVal/e1e2_3;
    }
    if (hexA1A2 == 0x31) {
        overlaps[0].ind[index] = expVal*(0.5*e1e2_2 - 0.5*e2*r2*(e1 - e2)*(e1 + e2) - e2*r2*(e2*(e1*e2*r2 - 0.5*e1 - 0.5*e2) + 0.5*(e1 - e2)*(e1 + e2)) + (e1 + e2)*(-0.5*e1*e2*r2 + 0.25*e1 + 0.25*e2))/e1e2_4;
        overlaps[1].ind[index] = expVal*r*(-1.5*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 2.25*e2_2 - 0.5*e2_3*r2)/(4.0*e1*e2_3 + 6.0*e1_2*e2_2 + 4.0*e1_3*e2 + 1.0*e1_4 + 1.0*e2_4);
        overlaps[2].ind[index] = expVal*(-0.25*e1*e2*r2 + 0.75*e1 + e2*r2*(-0.25*e1 + 0.5*e2) - 0.25*e2*r2*(e1 - e2) + 0.75*e2)/e1e2_4;
        overlaps[3].ind[index] = expVal*r*(0.125*e1 - 0.375*e2)/e1e2_4;
        overlaps[4].ind[index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x32) {
        overlaps[0].ind[index] = expVal*r*(-3.0*e1*e2_2 + 2.5*e1*e2_3*r2 + 0.75*e1_2*e2 + 1.5*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 - 1.5*e1_3*e2*r2 + 1.5*e1_3 - 2.25*e2_3 - 0.5*e2_4*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[1].ind[index] = expVal*(3.75*e1*e2 - 2.25*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 1.875*e1_2 + 0.75*e1_3*r2 + 1.875*e2_2 + 2.25*e2_3*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[2].ind[index] = expVal*r*(-0.75*e1*e2 + 1.5*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.5*e1_2 - 2.25*e2_2 - 0.25*e2_3*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[3].ind[index] = expVal*(-0.25*e1*e2*r2 + 0.625*e1 + 0.125*e1_2*r2 + 0.25*e2*r2*(-e1 + e2) - 0.5*e2*r2*(0.5*e1 - 0.25*e2) + 0.625*e2)/e1e2_5;
        overlaps[4].ind[index] = expVal*r*(0.125*e1 - 0.1875*e2)/e1e2_5;
        overlaps[5].ind[index] = 0.03125*expVal/e1e2_5;
    }
    if (hexA1A2 == 0x33) {
        overlaps[0].ind[index] = expVal*(5.625*e1*e2_2 - 2.25*e1*e2_3*r2 - 1.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 9.0*e1_2*e2_2*r2 + 3.0*e1_2*e2_3*r4 - 2.25*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 - 1.0*e1_3*e2_3*r6 + 1.875*e1_3 - 1.5*e1_4*e2*r4 + 2.25*e1_4*r2 + 1.875*e2_3 + 2.25*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[1].ind[index] = expVal*r*(-5.625*e1*e2_2 + 6.0*e1*e2_3*r2 + 5.625*e1_2*e2 - 1.5*e1_2*e2_3*r4 - 6.0*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 5.625*e1_3 + 0.75*e1_4*r2 - 5.625*e2_3 - 0.75*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[2].ind[index] = expVal*(5.625*e1*e2 - 4.5*e1*e2_2*r2 - 0.75*e1*e2_3*r4 - 4.5*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 2.8125*e1_2 - 0.75*e1_3*e2*r4 + 2.25*e1_3*r2 + 2.8125*e2_2 + 2.25*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[3].ind[index] = expVal*r*(1.125*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.125*e1_3*r2 - 1.875*e2_2 - 0.125*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[4].ind[index] = expVal*(-0.1875*e1*e2*r2 + 0.46875*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.1875*e1 + 0.125*e2) - 0.5*e2*r2*(0.375*e1 - 0.125*e2) + 0.46875*e2)/e1e2_6;
        overlaps[5].ind[index] = 0.09375*expVal*r*(e1 - e2)/e1e2_6;
        overlaps[6].ind[index] = 0.015625*expVal/e1e2_6;
    }
    if (hexA1A2 == 0x34) {
        overlaps[0].ind[index] = expVal*r*(-9.375*e1*e2_3 + 7.5*e1*e2_4*r2 + 5.625*e1_2*e2_2 + 3.75*e1_2*e2_3*r2 - 3.0*e1_2*e2_4*r4 + 16.875*e1_3*e2 - 15.0*e1_3*e2_2*r2 + 3.0*e1_3*e2_3*r4 - 7.5*e1_4*e2*r2 + 4.5*e1_4*e2_2*r4 - 1.0*e1_4*e2_3*r6 + 7.5*e1_4 - 1.5*e1_5*e2*r4 + 3.0*e1_5*r2 - 5.625*e2_4 - 0.75*e2_5*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[1].ind[index] = expVal*(19.6875*e1*e2_2 - 11.25*e1*e2_3*r2 - 3.0*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 10.5*e1_2*e2_3*r4 + 4.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 8.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 11.25*e1_4*r2 + 0.75*e1_5*r4 + 6.5625*e2_3 + 5.625*e2_4*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[2].ind[index] = expVal*r*(-5.625*e1*e2_2 + 8.25*e1*e2_3*r2 + 14.0625*e1_2*e2 - 4.5*e1_2*e2_2*r2 - 1.5*e1_2*e2_3*r4 - 10.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 11.25*e1_3 - 0.75*e1_4*e2*r4 + 3.0*e1_4*r2 - 8.4375*e2_3 - 0.75*e2_4*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[3].ind[index] = expVal*(6.5625*e1*e2 - 5.625*e1*e2_2*r2 - 0.5*e1*e2_3*r4 - 3.75*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.5*e1_3*e2*r4 + 3.75*e1_3*r2 + 0.125*e1_4*r4 + 3.28125*e2_2 + 1.875*e2_3*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[4].ind[index] = expVal*r*(0.46875*e1*e2 + 0.75*e1*e2_2*r2 - 1.125*e1_2*e2*r2 + 1.875*e1_2 + 0.25*e1_3*r2 - 1.40625*e2_2 - 0.0625*e2_3*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[5].ind[index] = expVal*(-0.125*e1*e2*r2 + 0.328125*e1 + 0.1875*e1_2*r2 + e2*r2*(-0.125*e1 + 0.0625*e2) - 0.5*e2*r2*(0.25*e1 - 0.0625*e2) + 0.328125*e2)/e1e2_7;
        overlaps[6].ind[index] = expVal*r*(0.0625*e1 - 0.046875*e2)/e1e2_7;
        overlaps[7].ind[index] = 0.0078125*expVal/e1e2_7;
    }
    if (hexA1A2 == 0x40) {
        overlaps[0].ind[index] = expVal*(0.5*e1e2_2 + 1.0*e2_2*r2*(e1 + e2) + e2_2*r2*(1.5*e1 + 1.5*e2 + e2_2*r2) + (e1 + e2)*(0.25*e1 + 0.25*e2 + 0.5*e2_2*r2))/e1e2_4;
        overlaps[1].ind[index] = e2*expVal*r*(-3.0*e1 - 3.0*e2 - 2.0*e2_2*r2)/e1e2_4;
        overlaps[2].ind[index] = expVal*(0.75*e1 + 0.75*e2 + 1.5*e2_2*r2)/e1e2_4;
        overlaps[3].ind[index] = -0.5*e2*expVal*r/e1e2_4;
        overlaps[4].ind[index] = 0.0625*expVal/e1e2_4;
    }
    if (hexA1A2 == 0x41) {
        overlaps[0].ind[index] = expVal*r*(-5.25*e1*e2_2 + 1.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 - 1.5*e1_2*e2 + 3.0*e1_2*e2_2*r2 + 0.75*e1_3 - 3.0*e2_3 - 2.0*e2_4*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[1].ind[index] = expVal*(3.75*e1*e2 + 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 3.0*e1_2*e2*r2 + 1.875*e1_2 + 1.875*e2_2 + 4.5*e2_3*r2 + 0.5*e2_4*r4)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[2].ind[index] = expVal*r*(-2.25*e1*e2 + 1.5*e1*e2_2*r2 + 0.75*e1_2 - 3.0*e2_2 - 1.0*e2_3*r2)/(5.0*e1*e2_4 + 10.0*e1_2*e2_3 + 10.0*e1_3*e2_2 + 5.0*e1_4*e2 + 1.0*e1_5 + 1.0*e2_5);
        overlaps[3].ind[index] = expVal*(-0.125*e1*e2*r2 + 0.625*e1 + 0.5*e2*r2*(-0.25*e1 + 0.5*e2) + e2*r2*(-0.125*e1 + 0.375*e2) - 0.125*e2*r2*(e1 - e2) + 0.625*e2)/e1e2_5;
        overlaps[4].ind[index] = expVal*r*(0.0625*e1 - 0.25*e2)/e1e2_5;
        overlaps[5].ind[index] = 0.03125*expVal/e1e2_5;
    }
    if (hexA1A2 == 0x42) {
        overlaps[0].ind[index] = expVal*(5.625*e1*e2_2 + 3.0*e1*e2_3*r2 - 3.5*e1*e2_4*r4 + 5.625*e1_2*e2 - 6.75*e1_2*e2_2*r2 - 1.0*e1_2*e2_3*r4 + 1.0*e1_2*e2_4*r6 - 4.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 1.875*e1_3 + 0.75*e1_4*r2 + 1.875*e2_3 + 4.5*e2_4*r2 + 0.5*e2_5*r4)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[1].ind[index] = expVal*r*(-11.25*e1*e2_2 + 6.0*e1*e2_3*r2 + 1.0*e1*e2_4*r4 + 6.0*e1_2*e2_2*r2 - 2.0*e1_2*e2_3*r4 - 3.0*e1_3*e2*r2 + 3.75*e1_3 - 7.5*e2_3 - 3.0*e2_4*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[2].ind[index] = expVal*(5.625*e1*e2 - 1.5*e1*e2_2*r2 - 2.0*e1*e2_3*r4 - 5.25*e1_2*e2*r2 + 1.5*e1_2*e2_2*r4 + 2.8125*e1_2 + 0.75*e1_3*r2 + 2.8125*e2_2 + 4.5*e2_3*r2 + 0.25*e2_4*r4)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[3].ind[index] = expVal*r*(-1.25*e1*e2 + 1.5*e1*e2_2*r2 - 0.5*e1_2*e2*r2 + 1.25*e1_2 - 2.5*e2_2 - 0.5*e2_3*r2)/(6.0*e1*e2_5 + 15.0*e1_2*e2_4 + 20.0*e1_3*e2_3 + 15.0*e1_4*e2_2 + 6.0*e1_5*e2 + 1.0*e1_6 + 1.0*e2_6);
        overlaps[4].ind[index] = expVal*(-0.125*e1*e2*r2 + 0.46875*e1 + 0.0625*e1_2*r2 + 0.125*e2*r2*(-e1 + e2) + e2*r2*(-0.125*e1 + 0.1875*e2) - 0.25*e2*r2*(0.5*e1 - 0.25*e2) + 0.46875*e2)/e1e2_6;
        overlaps[5].ind[index] = expVal*r*(0.0625*e1 - 0.125*e2)/e1e2_6;
        overlaps[6].ind[index] = 0.015625*expVal/e1e2_6;
    }
    if (hexA1A2 == 0x43) {
        overlaps[0].ind[index] = expVal*r*(-16.875*e1*e2_3 + 7.5*e1*e2_4*r2 + 1.5*e1*e2_5*r4 - 5.625*e1_2*e2_2 + 15.0*e1_2*e2_3*r2 - 4.5*e1_2*e2_4*r4 + 9.375*e1_3*e2 - 3.75*e1_3*e2_2*r2 - 3.0*e1_3*e2_3*r4 + 1.0*e1_3*e2_4*r6 - 7.5*e1_4*e2*r2 + 3.0*e1_4*e2_2*r4 + 5.625*e1_4 + 0.75*e1_5*r2 - 7.5*e2_4 - 3.0*e2_5*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[1].ind[index] = expVal*(19.6875*e1*e2_2 - 8.25*e1*e2_4*r4 + 19.6875*e1_2*e2 - 28.125*e1_2*e2_2*r2 + 4.5*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 10.5*e1_3*e2_2*r4 - 2.0*e1_3*e2_3*r6 + 6.5625*e1_3 - 3.0*e1_4*e2*r4 + 5.625*e1_4*r2 + 6.5625*e2_3 + 11.25*e2_4*r2 + 0.75*e2_5*r4)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[2].ind[index] = expVal*r*(-14.0625*e1*e2_2 + 10.5*e1*e2_3*r2 + 0.75*e1*e2_4*r4 + 5.625*e1_2*e2 + 4.5*e1_2*e2_2*r2 - 3.0*e1_2*e2_3*r4 - 8.25*e1_3*e2*r2 + 1.5*e1_3*e2_2*r4 + 8.4375*e1_3 + 0.75*e1_4*r2 - 11.25*e2_3 - 3.0*e2_4*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[3].ind[index] = expVal*(6.5625*e1*e2 - 3.75*e1*e2_2*r2 - 1.5*e1*e2_3*r4 - 5.625*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 0.5*e1_3*e2*r4 + 1.875*e1_3*r2 + 3.28125*e2_2 + 3.75*e2_3*r2 + 0.125*e2_4*r4)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[4].ind[index] = expVal*r*(-0.46875*e1*e2 + 1.125*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.40625*e1_2 + 0.0625*e1_3*r2 - 1.875*e2_2 - 0.25*e2_3*r2)/(7.0*e1*e2_6 + 21.0*e1_2*e2_5 + 35.0*e1_3*e2_4 + 35.0*e1_4*e2_3 + 21.0*e1_5*e2_2 + 7.0*e1_6*e2 + 1.0*e1_7 + 1.0*e2_7);
        overlaps[5].ind[index] = expVal*(-0.09375*e1*e2*r2 + 0.328125*e1 + 0.09375*e1_2*r2 + 0.09375*e2*r2*(-e1 + e2) + 0.5*e2*r2*(-0.1875*e1 + 0.125*e2) - 0.25*e2*r2*(0.375*e1 - 0.125*e2) + 0.328125*e2)/e1e2_7;
        overlaps[6].ind[index] = expVal*r*(0.046875*e1 - 0.0625*e2)/e1e2_7;
        overlaps[7].ind[index] = 0.0078125*expVal/e1e2_7;
    }
    // if (hexA1A2 == 0x44) {
    //     overlaps[0].ind[index] = expVal*(26.25*e1*e2_3 + 3.75*e1*e2_4*r2 - 10.5*e1*e2_5*r4 + 39.375*e1_2*e2_2 - 45.0*e1_2*e2_3*r2 + 3.75*e1_2*e2_4*r4 + 3.0*e1_2*e2_5*r6 + 26.25*e1_3*e2 - 45.0*e1_3*e2_2*r2 + 30.0*e1_3*e2_3*r4 - 5.0*e1_3*e2_4*r6 + 3.75*e1_4*e2*r2 + 3.75*e1_4*e2_2*r4 - 5.0*e1_4*e2_3*r6 + 1.0*e1_4*e2_4*r8 + 6.5625*e1_4 - 10.5*e1_5*e2*r4 + 3.0*e1_5*e2_2*r6 + 11.25*e1_5*r2 + 0.75*e1_6*r4 + 6.5625*e2_4 + 11.25*e2_5*r2 + 0.75*e2_6*r4)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[1].ind[index] = expVal*r*(-52.5*e1*e2_3 + 30.0*e1*e2_4*r2 + 3.0*e1*e2_5*r4 + 37.5*e1_2*e2_3*r2 - 15.0*e1_2*e2_4*r4 + 52.5*e1_3*e2 - 37.5*e1_3*e2_2*r2 + 2.0*e1_3*e2_4*r6 - 30.0*e1_4*e2*r2 + 15.0*e1_4*e2_2*r4 - 2.0*e1_4*e2_3*r6 + 26.25*e1_4 - 3.0*e1_5*e2*r4 + 7.5*e1_5*r2 - 26.25*e2_4 - 7.5*e2_5*r2)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[2].ind[index] = expVal*(39.375*e1*e2_2 - 11.25*e1*e2_3*r2 - 11.25*e1*e2_4*r4 + 39.375*e1_2*e2 - 56.25*e1_2*e2_2*r2 + 15.0*e1_2*e2_3*r4 + 1.5*e1_2*e2_4*r6 - 11.25*e1_3*e2*r2 + 15.0*e1_3*e2_2*r4 - 4.0*e1_3*e2_3*r6 + 13.125*e1_3 - 11.25*e1_4*e2*r4 + 1.5*e1_4*e2_2*r6 + 16.875*e1_4*r2 + 0.75*e1_5*r4 + 13.125*e2_3 + 16.875*e2_4*r2 + 0.75*e2_5*r4)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[3].ind[index] = expVal*r*(-13.125*e1*e2_2 + 12.5*e1*e2_3*r2 + 0.5*e1*e2_4*r4 + 13.125*e1_2*e2 - 3.0*e1_2*e2_3*r4 - 12.5*e1_3*e2*r2 + 3.0*e1_3*e2_2*r4 + 13.125*e1_3 - 0.5*e1_4*e2*r4 + 2.5*e1_4*r2 - 13.125*e2_3 - 2.5*e2_4*r2)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[4].ind[index] = expVal*(6.5625*e1*e2 - 4.6875*e1*e2_2*r2 - 1.0*e1*e2_3*r4 - 4.6875*e1_2*e2*r2 + 2.25*e1_2*e2_2*r4 + 3.28125*e1_2 - 1.0*e1_3*e2*r4 + 2.8125*e1_3*r2 + 0.0625*e1_4*r4 + 3.28125*e2_2 + 2.8125*e2_3*r2 + 0.0625*e2_4*r4)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[5].ind[index] = expVal*r*(0.75*e1*e2_2*r2 - 0.75*e1_2*e2*r2 + 1.3125*e1_2 + 0.125*e1_3*r2 - 1.3125*e2_2 - 0.125*e2_3*r2)/(8.0*e1*e2_7 + 28.0*e1_2*e2_6 + 56.0*e1_3*e2_5 + 70.0*e1_4*e2_4 + 56.0*e1_5*e2_3 + 28.0*e1_6*e2_2 + 8.0*e1_7*e2 + 1.0*e1_8 + 1.0*e2_8);
    //     overlaps[6].ind[index] = expVal*(-0.0625*e1*e2*r2 + 0.21875*e1 + 0.09375*e1_2*r2 + 0.5*e2*r2*(-0.125*e1 + 0.0625*e2) + e2*r2*(-0.0625*e1 + 0.046875*e2) - 0.25*e2*r2*(0.25*e1 - 0.0625*e2) + 0.21875*e2)/e1e2_8;
    //     overlaps[7].ind[index] = 0.03125*expVal*r*(e1 - e2)/e1e2_8;
    //     overlaps[8].ind[index] = 0.00390625*expVal/e1e2_8;
    // }
}

static void fill_overlaps(
    struct spatial::Vector *overlaps,
    const Gaussian3D &g1, const Gaussian3D &g2) {
    // spatial::Vector r = g1.position() - g2.position();
    // fp_t e1 = g1.orbital_exponent();
    // fp_t e2 = g2.orbital_exponent();
    // fp_t e_x = exp(-e1*e2*r.x*r.x/(e1 + e2));
    // fp_t e_y = exp(-e1*e2*r.y*r.y/(e1 + e2));
    // fp_t e_z = exp(-e1*e2*r.z*r.z/(e1 + e2));
    Gaussian1D g1x = g1.get_x();
    Gaussian1D g1y = g1.get_y();
    Gaussian1D g1z = g1.get_z();
    Gaussian1D g2x = g2.get_x();
    Gaussian1D g2y = g2.get_y();
    Gaussian1D g2z = g2.get_z();
    // int a1x = g1x.angular(), a2x = g2x.angular();
    // int a1y = g1y.angular(), a2y = g2y.angular();
    // int a1z = g1z.angular(), a2z = g2z.angular();
    fill_overlap_coefficients(overlaps, 1, g1x, g2x);
    fill_overlap_coefficients(overlaps, 2, g1y, g2y);
    fill_overlap_coefficients(overlaps, 3, g1z, g2z);
}

/* Obtain the overlap coefficients between two 3D Gaussians.

Refer to the section "Overlap Integrals" from Joshua Goings' blog post
here: https://joshuagoings.com/2017/04/28/integrals/.
*/
double overlap(const Gaussian3D &g1, const Gaussian3D &g2) {
    return g1.amplitude()*g2.amplitude() *
        overlap1d(g1.get_gaussian1d(0), g2.get_gaussian1d(0))
        * overlap1d(g1.get_gaussian1d(1), g2.get_gaussian1d(1))
        * overlap1d(g1.get_gaussian1d(2), g2.get_gaussian1d(2));
}

/* Compute the kinetic energy integral of two 3D Gaussians.

Refer to the section "Kinetic energy integrals" from Joshua Goings'
article: https://joshuagoings.com/2017/04/28/integrals/.
*/
double kinetic(const Gaussian3D &g1, const Gaussian3D &g2) {
    fp_t amplitude1 = g1.amplitude();
    fp_t amplitude2 = g2.amplitude();
    Gaussian1D g1x = g1.get_gaussian1d(0);
    Gaussian1D g1y = g1.get_gaussian1d(1);
    Gaussian1D g1z = g1.get_gaussian1d(2);
    Gaussian1D g2x = g2.get_gaussian1d(0);
    Gaussian1D g2y = g2.get_gaussian1d(1);
    Gaussian1D g2z = g2.get_gaussian1d(2);
    return -0.5*(
            laplacian1d(g1x, g2x)*overlap1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*laplacian1d(g1y, g2y)*overlap1d(g1z, g2z)
            + overlap1d(g1x, g2x)*overlap1d(g1y, g2y)*laplacian1d(g1z, g2z)
            )*amplitude1*amplitude2;
}

/* Obtain the integral of the product of two 3D Gaussians with a Coulomb
potential that represents a single nuclear charge. Refer to the section
"Nuclear attraction integrals" from Joshua Goings' blog post:
https://joshuagoings.com/2017/04/28/integrals/.
*/
double nuclear_single_charge(const Gaussian3D &g, const Gaussian3D &h,
                             struct Vector r) {
    Gaussian1D gx = g.get_x();
    Gaussian1D gy = g.get_y();
    Gaussian1D gz = g.get_z();
    Gaussian1D hx = h.get_x();
    Gaussian1D hy = h.get_y();
    Gaussian1D hz = h.get_z();
    Vector r2 = product_center(g, h);
    Vector r12 = r2 - r;
    fp_t orb_exp = g.orbital_exponent() + h.orbital_exponent();
    fp_t val = 0.0;
    // fp_t bf[16];
    // int ang_max =  gx.angular() + hx.angular() 
    //     + gy.angular() + hy.angular()
    //     + gz.angular() + hz.angular();
    // beylkin_sharma::compute_bf_table(
    //     bf, orb_exp*dot(r12, r12), ang_max);
    // Vector overlaps[8];
    // fill_overlaps(overlaps, g, h);
    for (int i = 0; i <= gx.angular() + hx.angular(); i++) {
        for (int j = 0; j <= gy.angular() + hy.angular(); j++) {
            for (int k = 0; k <= gz.angular() + hz.angular(); k++) {
                val += overlap_coefficient(i, gx, hx)
                       * overlap_coefficient(j, gy, hy)
                       * overlap_coefficient(k, gz, hz)
                       * coulomb_coefficient(i, j, k, 0,
                                             orb_exp, r12);
            }
        }
    }
    return -g.amplitude()*h.amplitude()*2.0*PI*val/orb_exp;

}

static void sort_indices(
    int &new0, int &new1, int &new2,
    int old0, int old1, int old2) {
    if (old2 > old1)
        std::swap(old2, old1);
    // 1 3 2
    if (old1 > old0)
        std::swap(old0, old1);
    // 3 1 2
    if (old2 > old1)
        std::swap(old2, old1);
    new0 = old0;
    new1 = old1;
    new2 = old2;
}

/* Compute the repulsion-exchange integrals.
Refer to the section "Two electron repulsion integrals" from 
Joshua Goings' blog post:
    https://joshuagoings.com/2017/04/28/integrals/.
*/
double repulsion(const Gaussian3D &g1,
                 const Gaussian3D &h1,
                 const Gaussian3D &g2,
                 const Gaussian3D &h2) {
    fp_t amplitude = g1.amplitude()*g2.amplitude()
                        *h1.amplitude()*h2.amplitude();
    if (amplitude == 0.0)
        return 0.0;
    fp_t val = 0.0;
    Gaussian1D g1x = g1.get_x(), g1y = g1.get_y(), g1z = g1.get_z();
    Gaussian1D h1x = h1.get_x(), h1y = h1.get_y(), h1z = h1.get_z();
    Gaussian1D g2x = g2.get_x(), g2y = g2.get_y(), g2z = g2.get_z();
    Gaussian1D h2x = h2.get_x(), h2y = h2.get_y(), h2z = h2.get_z();
    fp_t orb_exp1 = g1.orbital_exponent() + h1.orbital_exponent();
    fp_t orb_exp2 = g2.orbital_exponent() + h2.orbital_exponent();
    fp_t orb_exp = orb_exp1*orb_exp2/(orb_exp1 + orb_exp2);
    int ang_x1_max = g1x.angular() + h1x.angular();
    int ang_x2_max = g2x.angular() + h2x.angular();
    int ang_y1_max = g1y.angular() + h1y.angular();
    int ang_y2_max = g2y.angular() + h2y.angular();
    int ang_z1_max = g1z.angular() + h1z.angular();
    int ang_z2_max = g2z.angular() + h2z.angular();
    int ang_max = ang_x1_max + ang_y1_max + ang_z1_max 
        + ang_x2_max + ang_y2_max + ang_z2_max;
    Vector r12 = product_center(g1, h1) - product_center(g2, h2);
    Vector overlaps1[8], overlaps2[8];
    fill_overlaps(overlaps1, g1, h1);
    fill_overlaps(overlaps2, g2, h2);
    fp_t bf[16];
    beylkin_sharma::compute_bf_table(
        bf, orb_exp*dot(r12, r12), ang_max);
    for (int ix = 0; ix <= ang_x1_max; ix++) {
        fp_t o1x = overlaps1[ix].x;
        for (int iy = 0; iy <= ang_y1_max; iy++) {
            fp_t o1y = overlaps1[iy].y;
            for (int iz = 0; iz <= ang_z1_max; iz++) {
                fp_t o1z = overlaps1[iz].z;
                for (int jx = 0; jx <= ang_x2_max; jx++) {
                    fp_t o2x = overlaps2[jx].x;
                    for (int jy = 0; jy <= ang_y2_max; jy++) {
                        fp_t o2y = overlaps2[jy].y;
                        for (int jz = 0; jz <= ang_z2_max; jz++) {
                            fp_t o2z = overlaps2[jz].z;
                            // fp_t o1x = overlap_coefficient(ix, g1x, h1x);
                            // fp_t o1y = overlap_coefficient(iy, g1y, h1y);
                            // fp_t o1z = overlap_coefficient(iz, g1z, h1z);
                            // fp_t o2x = overlap_coefficient(jx, g2x, h2x);
                            // fp_t o2y = overlap_coefficient(jy, g2y, h2y);
                            // fp_t o2z = overlap_coefficient(jz, g2z, h2z);
                            fp_t overlaps = o1x * o1y * o1z * o2x * o2y * o2z;
                            fp_t cc = coulomb_coefficient(
                                bf,
                                ix + jx, iy + jy, iz + jz, 0, orb_exp, r12);
                            fp_t factor = (((jx +jy + jz) % 2) == 1)? -1.0: 1.0;
                            val += factor * overlaps * cc;
                        }
                    }
                }
            }
        }
    }
    return amplitude*val*(2.0*pow(PI, (5.0/2.0))) 
        / (orb_exp1*orb_exp2*std::sqrt(orb_exp1 + orb_exp2));

}
