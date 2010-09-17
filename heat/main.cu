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

__global__ void do_deriv_opt(float *p, int n, float *from, float *source, float inv_dx)
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float cache[256];

  if (idx < n) {
    cache[tid] = from[idx];
  }

  __syncthreads();

  if (idx < n) {
    float s = source[idx];
    float accum;
   
    if (tid == 0) {
      int i_m = idx - 1;
      if (i_m == -1) i_m = n-1;
      accum = from[i_m];
    }
    else
      accum = cache[tid-1];

    if (tid == 255 || idx == (n-1)) {
      int i_p = idx + 1;    
      if (i_p == n)  i_p = 0;
      accum += from[i_p];
    }
    else
      accum += cache[tid+1];

    accum -= 2*cache[tid];
    accum *= inv_dx;
    accum += s;

    p[idx] = accum;
  }
}

__global__ void do_deriv(float *p, int n, float *from, float *source, float inv_dx)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    int i_m = idx - 1;
    int i_p = idx + 1;    
    if (i_m == -1) i_m = n-1;
    if (i_p == n)  i_p = 0;
    float accum = source[idx] + (from [i_m] - 2 * from[idx] + from[i_p]) * inv_dx;
    p[idx] = accum;
  }
}

__global__ void add_mult(float *p, int n, float *from, float scale)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    p[idx] += from[idx] * scale;
}

__global__ void set_to_zero(float *p, int n)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    p[idx] = 0;
}

__global__ void set_in_range(float *p, int n, int start, int end)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) {
    if (idx >= start && idx <= end)
      p[idx] = 1;
    else
      p[idx] = 0;
  }
}


int main_gpu2(int argc, const char **argv)
{
  int n = 1024 * 1024;
  float dt = 0.1f;
  float dx = 0.5f;

  DeviceArray1D phi, dphidt, source;
  HostArray1D hphi;

  phi.allocate(n,0);
  dphidt.allocate(n,0);
  source.allocate(n,0);
  hphi.allocate(n,0);

  set_to_zero<<<(n+255) / 256, 256>>>(&phi.at(0), n);
  set_in_range<<<(n+255)/256, 256>>>(&source.at(0), n, n/2, n/2 + n/4);


  cpu_timer timer;
  timer.start();
  for (int step = 0; step < 100; step++) {
    do_deriv_opt<<<(n+255)/256, 256>>>(&dphidt.at(0), n, &phi.at(0), &source.at(0), 1.0f/dx);
    add_mult<<<(n+255)/256, 256>>>(&phi.at(0), n, &dphidt.at(0), dt);

  }
  timer.stop();
  printf("Elapsed: %f\n", timer.elapsed_ms());


  return 0;
}

int main_gpu1(int argc, const char **argv)
{
  int n = 1024 * 1024;
  float dt = 0.1f;
  float dx = 0.5f;

  DeviceArray1D phi, dphidt, source;
  HostArray1D hphi;

  phi.allocate(n,0);
  dphidt.allocate(n,0);
  source.allocate(n,0);
  hphi.allocate(n,0);

  set_to_zero<<<(n+255) / 256, 256>>>(&phi.at(0), n);
  set_in_range<<<(n+255)/256, 256>>>(&source.at(0), n, n/2, n/2 + n/4);

  cpu_timer timer;
  timer.start();
  for (int step = 0; step < 100; step++) {
    do_deriv<<<(n+255)/256, 256>>>(&dphidt.at(0), n, &phi.at(0), &source.at(0), 1.0f/dx);
    add_mult<<<(n+255)/256, 256>>>(&phi.at(0), n, &dphidt.at(0), dt);
  }
  timer.stop();
  printf("Elapsed: %f\n", timer.elapsed_ms());

  return 0;
}

int main_cpu(int argc, const char **argv)
{
  int n = 1024 * 1024;
  float dt = 0.1f;
  float dx = 0.5f;

  HostArray1D phi, dphidt, source;

  phi.allocate(n,0);
  dphidt.allocate(n,0);
  source.allocate(n,0);

  int i;
  for (i=0; i < n; i++)
    phi.at(i) = 0;
  for (i=0; i < n; i++)
    source.at(i) = 0;
  for (i=n/2; i <= n/2 + n/4; i++)
    source.at(i) = 1;


  cpu_timer timer;
  timer.start();
  for (int step = 0; step < 100; step++) {
    for (i=0 ; i < n; i++) {
      int i_m = (i-1+n)%n;
      int i_p = (i+1)%n;
      dphidt.at(i) = (phi.at(i_m) - 2*phi.at(i) + phi.at(i_p))/dx + source.at(i);
    }

    for(i=0; i < n; i++)
      phi.at(i) += dt * dphidt.at(i);
  }
  timer.stop();

  printf("Elapsed: %f\n", timer.elapsed_ms());
  return 0;
}


int main_metaprog(int argc, const char **argv)
{
  int n = 1024 * 1024;
  float dt = 0.1f;
  float dx = 0.5f;

  DeviceArray1D phi, dphidt, source;
  HostArray1D hphi;

  phi.allocate(n,0);
  dphidt.allocate(n,0);
  source.allocate(n,0);
  hphi.allocate(n,0);

  phi = constant(0);
  source = inrange(identity(), constant(n/2), constant(n/2 + n/4));

  cpu_timer timer;
  timer.start();
  for (int step = 0; step < 100; step++) {
    dphidt = (constant(1/dx) * (phi[-1] - constant(2) * phi[0] + phi[1]) + source[0]);
    phi = phi[0] + constant(dt) * dphidt[0];

  }
  timer.stop();
  printf("Elapsed: %f\n", timer.elapsed_ms());

  return 0;
}

int main(int argc, const char **argv)
{
  if (argc == 1) {
    printf("usage: run [cpu|gpu1|gpu2|meta]\n");
    exit(-1);
  }

  if (strcmp(argv[1], "cpu")==0)
    return main_cpu(argc, argv);
  if (strcmp(argv[1], "gpu1")==0)
    return main_gpu1(argc, argv);
  if (strcmp(argv[1], "gpu2")==0)
    return main_gpu2(argc, argv);
  if (strcmp(argv[1], "meta")==0)
    return main_metaprog(argc, argv);
}

