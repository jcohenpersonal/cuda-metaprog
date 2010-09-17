/*
 *  Copyright 2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "operators.h"
#include "timer.h"
#include <vector>
#include <thrust/inner_product.h>

#ifdef _WIN32
 #pragma warning (disable: 4244)
 #pragma warning (disable: 4503)
#endif


int n_levels(int n)
{
  int val = 1;
  while (n > 4) {
    val++;
    n /= 2;
  }

  return val;
}

void update_bc(DeviceArray1D &U, int level)
{
  int n = U._size;

  // at all levels but the finest, use 0 boundary conditions
  double left_dirichelet = level == 0 ? 1 : 0;
  double right_dirichelet = level == 0 ? 0 : 0;

  // enforce dirichelet conditions at the half-grid points, U[-.5] and U[n-.5], hence the extrapolation
  U(-1) = -U.read(0)   + constant(2.0 * left_dirichelet);
  U(n)  = -U.read(n-1) + constant(2.0 * right_dirichelet);
}

void relax(DeviceArray1D &U, DeviceArray1D &B, int nu, double h, int level)
{
  int n = U._size;

  for (int i=0; i < nu; i++) {

    update_bc(U, level);

    // read(start, end, step), where [start, end] is inclusive

    // red update
    // fortran style:
//  U(0,n-2,2) = constant(.5) * (constant(-h*h) * B.read(0, n-2, 2) + U.read(-1, n-3, 2) + U.read(1, n-1, 2));
    // stencil-style:
//  U(0,n-2,2) = constant(.5) * (constant(-h*h) * B[0] + U[-1] + U[1]);

    // properly accounting for boundary conditions
    // see: http://code.google.com/p/opencurrent/wiki/EnforcingBoundaryConditionsInMultigrid
    U(0,n-2,2) = U.read(0,n-2,2) + (constant(-h*h) * B[0] + U[-1] + U[1] - constant(2) * U[0]) / 
      (constant(2) + inrange(index(), constant(0), constant(0)) + inrange(index(), constant(n-1), constant(n-1)));

    // black update
    U(1,n-1,2) = U.read(1,n-1,2) + (constant(-h*h) * B[0] + U[-1] + U[1] - constant(2) * U[0]) / 
      (constant(2) + inrange(index(), constant(0), constant(0)) + inrange(index(), constant(n-1), constant(n-1)));
  }
}

double l2_norm(const DeviceArray1D &R)
{
  thrust::device_ptr<const double> first(&R.at(0));
  thrust::device_ptr<const double> last (&R.at(R._size));
  return sqrt(thrust::inner_product(first, last, first, (double)0)/R._size);
}

double integral(DeviceArray1D &R)
{
  thrust::device_ptr<double> first(&R.at(0));
  thrust::device_ptr<double> last (&R.at(R._size));

  return thrust::reduce(first, last);
}

void restrict_residual(DeviceArray1D &U_f, const DeviceArray1D &B_f, DeviceArray1D &R_f, DeviceArray1D &B_c, double h, int level)
{
  int n = R_f._size;

  update_bc(U_f, level);

  // residual
  R_f = B_f[0] - constant(1.0/(h*h)) * (U_f[1] + U_f[-1] - constant(2) * U_f[0]);

  // restrict to coarse RHS
  B_c = constant(.5) * (R_f.read(0, n-2, 2) + R_f.read(1, n-1, 2));
}

double residual(const DeviceArray1D &U, const DeviceArray1D &B, DeviceArray1D &R, double h)
{
  R = B[0] - constant(1.0/(h*h)) * (U[1] + U[-1] - constant(2) * U[0]);

  return l2_norm(R);
}

void prolong(DeviceArray1D &U_c, DeviceArray1D &U_f, int level)
{
  update_bc(U_c, level+1);

  int n_c = U_c._size; 
  int n_f = U_f._size; 
  
  // linear interpolation from coarse to fine:
  //
  // coarse: o   |   o   |   o   |   o
  // fine  :   x | x | x | x | x | x

  U_f(-1, n_f-1, 2) = U_f.read(-1, n_f-1, 2) + constant(.75) * U_c.read(-1, n_c-1) + constant(.25) * U_c.read(0, n_c);
  U_f( 0, n_f  , 2) = U_f.read( 0, n_f  , 2) + constant(.25) * U_c.read(-1, n_c-1) + constant(.75) * U_c.read(0, n_c);

  update_bc(U_f, level);
}


int n_from_level(int n, int l)
{
  return n >> l; 
}

double do_vcycle(
  std::vector<DeviceArray1D> &U, 
  std::vector<DeviceArray1D> &B, 
  std::vector<DeviceArray1D> &R,
  std::vector<double> &hx)
{
  int nu1 = 2;
  int nu2 = 2;

  int coarse_level = U.size()-1;
  int level=0;

  //relax(U[level], B[level], 100, hx[level]);

  // going down
  for(level = 0; level < coarse_level; level++) {
    relax(U[level], B[level], nu1, hx[level], level);

    restrict_residual(U[level], B[level], R[level], B[level+1], hx[level], level);

    U[level+1] = constant(0);
    update_bc(U[level+1], level+1);
  }

  relax(U[coarse_level], B[coarse_level], 10, hx[coarse_level], coarse_level);

  // going up
  for (level=coarse_level-1; level >= 0; level--) {
    prolong(U[level+1], U[level], level);
    relax(U[level], B[level], nu2, hx[level], level);
  }

  return residual(U[0], B[0], R[0], hx[0]);
} 

int main_metaprog(int argc, const char **argv)
{
  int n = 1024*1024*4;

  int nlevels = n_levels(n);
  printf("nlevels = %d\n", nlevels);
  int l;

  std::vector<DeviceArray1D> U(nlevels), B(nlevels), R(nlevels);
  std::vector<double> hx(nlevels);

  for (l=0; l < nlevels; l++) {
    int nl = n_from_level(n,l);
    hx[l] = 100.0 / nl;
    printf("size at level %d = %d (%f)\n", l, n_from_level(n, l), hx[l]);
    U[l].allocate(nl, 1);
    B[l].allocate(nl, 1);
    R[l].allocate(nl, 1);

    U[l](-1,nl) = constant(0);
    R[l](-1,nl) = constant(0);
    B[l](-1,nl) = constant(0);

    update_bc(U[l], l);
  }
 
  // solve with 0 RHS, randon initial guess
  U[0] = rand(10) - constant(.5);
  update_bc(U[0], 0);

  cpu_timer timer;
  timer.start();

  for (int step = 0; step < 10; step++) {

    double resid = do_vcycle(U, B, R, hx);
    printf("%d: residual %f\n", step, resid);

  }
  timer.stop();
  printf("Elapsed: %f\n", timer.elapsed_ms());

  return 0;
}

int main(int argc, const char **argv)
{
  return main_metaprog(argc, argv);
}

