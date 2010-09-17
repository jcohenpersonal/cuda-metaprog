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

#include <cstdio>



struct Range {
  int minx, maxx;

  Range(int _minx, int _maxx) :
    minx(_minx), maxx(_maxx) { } 

  bool contains(const Range &r) const {
    return r.minx >= minx && r.maxx <= maxx;
  } 

  Range shift(int shiftx) const {
    return Range(minx + shiftx, maxx + shiftx);
  }

  Range step(int stepx) const {
    return Range(minx*stepx, maxx*stepx);
  }


  void print() const { printf("(%d, %d)\n", minx, maxx); }
};


template<class OP, class PARM>
struct OpWithParm {

  struct DeviceParm {
    typename PARM::DeviceParm parm;

    DeviceParm() { }

    __device__ float exec(int i) const { return OP::exec(i, parm); }
  };

  PARM parm;
  OpWithParm(const PARM &p) : parm(p) { }
  bool validate(const Range &rng) const { return OP::validate(rng, parm); }

  DeviceParm kernel_data() const        { DeviceParm dparm; parm.set_kernel_data(dparm.parm); return dparm; }
};

template<class PARM>
struct LeafOp;

struct ArrayLookupParm;
class DeviceArray1D;

OpWithParm<LeafOp<ArrayLookupParm>, ArrayLookupParm>
read(const DeviceArray1D &g, int shift);


struct Array1D
{
  int _size;
  int _padding;
  size_t _total_bytes;
  bool _valid;
  float *_shifted_ptr;
  float *_ptr;

  Array1D() : _size(0), _padding(0), _shifted_ptr(0), _ptr(0), _total_bytes(0), _valid(false) { }

  Range range() const { return Range(0, _size-1); }

  float &at(int i) { return _shifted_ptr[i]; }
  const float &at(int i) const { return _shifted_ptr[i]; }
};

struct HostArray1D : public Array1D
{
  HostArray1D() {} 
  ~HostArray1D() {
    if (_ptr) cudaFreeHost(_ptr); 
  }

  bool allocate(int size, int padding) {
    _size = size;
    _padding = padding;
    _total_bytes = sizeof(float) * (_size + 2 * _padding);
    cudaMallocHost((void **)&_ptr, _total_bytes);
    _shifted_ptr = _ptr + _padding;
    _valid = true;
    return true;
  }
};

class DeviceArray1D;
template <typename T>
bool execute_expression(T functor, DeviceArray1D &dst);


struct DeviceArray1D : public Array1D
{
  DeviceArray1D() {}
  ~DeviceArray1D() {
    if (_ptr) cudaFree(_ptr); 
  }

  bool allocate(int size, int padding) {
    _size = size;
    _padding = padding;
    _total_bytes = sizeof(float) * (_size + 2 * _padding);
    cudaMalloc((void **)&_ptr, _total_bytes);
    _shifted_ptr = _ptr + _padding;
    _valid = true;
    return true;
  }

  template<typename T>
  DeviceArray1D &operator=(const T &t) { _valid = execute_expression(t, *this); return *this; } 

  OpWithParm<LeafOp<ArrayLookupParm>, ArrayLookupParm>
  operator[](int i) const;
};

bool copy(DeviceArray1D &to, const HostArray1D &from)
{
  if (to._total_bytes != from._total_bytes) return false;
  cudaMemcpy(to._ptr, from._ptr, to._total_bytes, cudaMemcpyHostToDevice);

  return true;
}

bool copy(HostArray1D &to, const DeviceArray1D &from)
{
  if (to._total_bytes != from._total_bytes) return false;
  cudaMemcpy(to._ptr, from._ptr, to._total_bytes, cudaMemcpyDeviceToHost);

  return true;
}

