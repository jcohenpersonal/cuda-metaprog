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

#include "array1d.h"
#include "timer.h"
#include <vector>
#include <numeric>

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

void update_bc(HostArray1D &U, int level)
{
  int n = U._size;

  // at all levels but the finest, use 0 boundary conditions
  double left_dirichelet = level == 0 ? 1 : 0;
  double right_dirichelet = level == 0 ? 0 : 0;

  // enforce dirichelet conditions at the half-grid points, U[-.5] and U[n-.5], hence the extrapolation
  U.at(-1) = -U.at(0)   + 2.0 * left_dirichelet;
  U.at(n)  = -U.at(n-1) + 2.0 * right_dirichelet;
}

void relax(HostArray1D &U, HostArray1D &B, int nu, double h, int level)
{
  int n = U._size;

  for (int i=0; i < nu; i++) {

    update_bc(U, level);

    // read(start, end, step), where [start, end] is inclusive

    // red update
    // fortran style:

    // properly accounting for boundary conditions
    // see: http://code.google.com/p/opencurrent/wiki/EnforcingBoundaryConditionsInMultigrid
    int j;

    for (j=0; j <= n-2; j += 2)
      U.at(j) += (-h*h * B.at(j) + U.at(j-1) + U.at(j+1) - 2.0f * U.at(j) ) / (2 + (j==0 ? 1 : j == n-1 ? 1 : 0) );

    // black update
    for (j=1; j <= n-1; j += 2)
      U.at(j) += (-h*h * B.at(j) + U.at(j-1) + U.at(j+1) - 2.0f * U.at(j) ) / (2 + (j==0 ? 1 : j == n-1 ? 1 : 0) ); 
  }
}

double l2_norm(const HostArray1D &R)
{
  return sqrt(std::inner_product(&R.at(0), &R.at(R._size), &R.at(0), (double)0)/R._size);
}

double integral(HostArray1D &R)
{

  return std::accumulate(&R.at(0), &R.at(R._size), (double)0);
}

void restrict_residual(HostArray1D &U_f, const HostArray1D &B_f, HostArray1D &R_f, HostArray1D &B_c, double h, int level)
{
  int n = R_f._size;

  update_bc(U_f, level);

  // residual
  int k;
  for (k=0; k < n; k++)
    R_f.at(k) = B_f.at(k) - 1.0f/(h*h) * (U_f.at(k+1) + U_f.at(k-1) - 2.0f * U_f.at(k));

  // restrict to coarse RHS
  for (k=0; k < B_c._size; k++)
    B_c.at(k) = .5f * (R_f.at(k*2) + R_f.at(k*2+1));
}

double residual(const HostArray1D &U, const HostArray1D &B, HostArray1D &R, double h)
{
  for (int k=0; k < R._size; k++)
    R.at(k) = B.at(k) - (1.0f/(h*h)) * (U.at(k+1) + U.at(k-1) - 2.0f * U.at(k));

  return l2_norm(R);
}

void prolong(HostArray1D &U_c, HostArray1D &U_f, int level)
{
  update_bc(U_c, level+1);

  int n_f = U_f._size; 
  
  // linear interpolation from coarse to fine:
  //
  // coarse: o   |   o   |   o   |   o
  // fine  :   x | x | x | x | x | x
  int k, kc;

  for (k=-1, kc=-1; k <= n_f-1; k += 2, kc++)
    U_f.at(k) += .75f * U_c.at(kc) + .25f * U_c.at(kc+1);

  for (k=0, kc=-1; k <= n_f; k += 2, kc++)
    U_f.at(k) += .25f * U_c.at(kc) + .75f * U_c.at(kc+1);

  update_bc(U_f, level);
}


int n_from_level(int n, int l)
{
  return n >> l; 
}

double do_vcycle(
  std::vector<HostArray1D> &U, 
  std::vector<HostArray1D> &B, 
  std::vector<HostArray1D> &R,
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

    for (int k=0; k < U[level+1]._size; k++)
      U[level+1].at(k) = 0;

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

int main_cpu(int argc, const char **argv)
{
  int n = 1024 * 1024 * 4;

  int nlevels = n_levels(n);
  printf("nlevels = %d\n", nlevels);
  int l;

  std::vector<HostArray1D> U(nlevels), B(nlevels), R(nlevels);
  std::vector<double> hx(nlevels);

  for (l=0; l < nlevels; l++) {
    int nl = n_from_level(n,l);
    hx[l] = 100.0 / nl;
    printf("size at level %d = %d (%f)\n", l, n_from_level(n, l), hx[l]);
    U[l].allocate(nl, 1);
    B[l].allocate(nl, 1);
    R[l].allocate(nl, 1);

    for (int k=-1; k <= nl; k++) {
      U[l].at(k) = 0;
      R[l].at(k) = 0;
      B[l].at(k) = 0;
    }

    update_bc(U[l], l);
  }
 
  // solve with 0 RHS, randon initial guess
  for (int k=0; k < U[0]._size; k++)
    U[0].at(k) = ((double)rand()) / ((double)RAND_MAX) - .5f; 
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
  return main_cpu(argc, argv);
}

