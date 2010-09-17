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

  int size(int stride=1) const { return 1 + ((maxx - minx) / stride); }

  bool contains(const Range &r) const {
    return r.minx >= minx && r.maxx <= maxx;
  } 

  Range shift(int shiftx) const {
    return Range(minx + shiftx, maxx + shiftx);
  }

  Range stride(int stridex) const {
    return Range(minx*stridex, maxx*stridex);
  }

  void print() const { printf("(%d, %d)\n", minx, maxx); }
};


template<class OP, class PARM>
struct OpWithParm {

  struct DeviceParm {
    typename PARM::DeviceParm parm;

    DeviceParm() { }

    __device__ double exec(int ordinal, int index) const { return OP::exec(ordinal, index, parm); }
  };

  PARM parm;
  OpWithParm(const PARM &p) : parm(p) { }
  bool validate(const Range &ord_rng, const Range &idx_rng) const { return OP::validate(ord_rng, idx_rng, parm); }

  DeviceParm kernel_data() const        { DeviceParm dparm; parm.set_kernel_data(dparm.parm); return dparm; }
};

template<class PARM>
struct LeafOp;

struct ArrayLookupIndexParm;
struct ArrayLookupOrdinalParm;
struct DeviceArray1D;


struct Array1D
{
  int _size;
  int _padding;
  size_t _total_bytes;
  bool _valid;
  double *_shifted_ptr;
  double *_ptr;

  Array1D() : _size(0), _padding(0), _shifted_ptr(0), _ptr(0), _total_bytes(0), _valid(false) { }

  Range valid_range() const { return Range(-_padding, _size-1+_padding); }

  double &at(int i) { return _shifted_ptr[i]; }
  const double &at(int i) const { return _shifted_ptr[i]; }
};

struct HostArray1D : public Array1D
{
  HostArray1D() {} 
  ~HostArray1D() {
    if (_ptr) free(_ptr); 
  }

  bool allocate(int size, int padding) {
    _size = size;
    _padding = padding;
    _total_bytes = sizeof(double) * (_size + 2 * _padding);
    _ptr = (double *)malloc(_total_bytes);
    _shifted_ptr = _ptr + _padding;
    _valid = true;
    return true;
  }
};

struct DeviceArray1D;
template <typename OP, typename PARM>
bool execute_expression(const OpWithParm<OP,PARM> &func, DeviceArray1D &dst, int ordinal_size, int idx_start, int idx_step);

struct DeviceSubArray1D;

struct DeviceArray1D : public Array1D
{
  DeviceArray1D() {}
  ~DeviceArray1D() {
    if (_ptr) cudaFree(_ptr); 
  }

  bool allocate(int size, int padding) {
    _size = size;
    _padding = padding;
    _total_bytes = sizeof(double) * (_size + 2 * _padding);
    cudaMalloc((void **)&_ptr, _total_bytes);
    _shifted_ptr = _ptr + _padding;
    _valid = true;
    return true;
  }

  template <typename OP, typename PARM>
  DeviceArray1D &operator=(const OpWithParm<OP,PARM> &t) { 
    _valid = execute_expression(t, *this, this->_size, 0, 1); return *this; 
  } 

  OpWithParm<LeafOp<ArrayLookupIndexParm>, ArrayLookupIndexParm>     operator[](int i) const;
  OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm> read(int minx, int maxx, int stride=1) const;
  OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm> read(int minx) const;

  DeviceSubArray1D operator()(int minx);
  DeviceSubArray1D operator()(int minx, int maxx, int stride=1);

};

struct DeviceSubArray1D : public Array1D
{
  DeviceArray1D &_array;
  Range          _sub;
  int            _stride;

  DeviceSubArray1D(DeviceArray1D &d, Range &r, int stride) : _array(d), _sub(r), _stride(stride) {
    _size = d._size;
    _padding = d._padding;
    _total_bytes = d._total_bytes;
    _valid = d._valid;;
    _shifted_ptr = d._shifted_ptr;;
    _ptr = d._ptr;
  }

  template <typename OP, typename PARM>
  DeviceSubArray1D &operator=(const OpWithParm<OP,PARM> &t) { 
    _valid = execute_expression(t, this->_array, this->_sub.size(this->_stride), this->_sub.minx, this->_stride); return *this; 
  } 

};

DeviceSubArray1D DeviceArray1D::operator()(int minx)
{
  return DeviceSubArray1D(*this, Range(minx, minx), 1);
}

DeviceSubArray1D DeviceArray1D::operator()(int minx, int maxx, int stride)
{
  return DeviceSubArray1D(*this, Range(minx, maxx), stride);
}

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

