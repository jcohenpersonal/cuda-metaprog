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

// contaioner class that holds 2 params which can be passed recursively to children operators

template<typename LPARM, typename RPARM>
struct ParmPair {

  struct DeviceParm {
    typename LPARM::DeviceParm left;
    typename RPARM::DeviceParm right;
  };


  LPARM left;
  RPARM right;
  ParmPair(const LPARM &l, const RPARM &r) : left(l), right(r) { } 

  void set_kernel_data(DeviceParm &dp) const {
    left.set_kernel_data(dp.left);
    right.set_kernel_data(dp.right);
  }
};


template<typename LPARM, typename MPARM, typename RPARM>
struct ParmTriple {

  struct DeviceParm {
    typename LPARM::DeviceParm left;
    typename MPARM::DeviceParm middle;
    typename RPARM::DeviceParm right;
  };


  LPARM left;
  MPARM middle;
  RPARM right;
  ParmTriple(const LPARM &l, const MPARM &m, const RPARM &r) : left(l), middle(m), right(r) { } 

  void set_kernel_data(DeviceParm &dp) const {
    left.set_kernel_data(dp.left);
    middle.set_kernel_data(dp.middle);
    right.set_kernel_data(dp.right);
  }
};

// Op classes have a method called exec(int i, Param p) that returns the value for loop index i
// and param p.  This might call p.value, or pass p down to their children
template<class LOP, class LPARM>
struct UnaryOp {
  static bool validate(const Range &rng, LPARM &p) {
    return LOP::validate(rng, p);
  } 

};

// Op classes have a method called exec(int i, Param p) that returns the value for loop index i
// and param p.  This might call p.value, or pass p down to their children
template<class LOP, class ROP, class LPARM, class RPARM>
struct BinaryOp {
  static bool validate(const Range &rng, const ParmPair<LPARM, RPARM> &p) {
    return LOP::validate(rng, p.left) && ROP::validate(rng, p.right);
  } 

};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct TernaryOp {
  static bool validate(const Range &rng, const ParmTriple<LPARM, MPARM, RPARM> &p) {
    return LOP::validate(rng, p.left) && MOP::validate(rng, p.middle), ROP::validate(rng, p.right);
  } 

};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct MedianOp : public TernaryOp<LOP, MOP, ROP, LPARM, MPARM, RPARM> {
  __device__ static float exec(int i, const typename ParmTriple<LPARM, MPARM, RPARM>::DeviceParm &p) {
    float a = LOP::exec(i, p.left);
    float b = MOP::exec(i, p.middle);
    float c = ROP::exec(i, p.right); 
    return a < b ? 
      (b < c ? b : (a < c ? c : a)) : 
      (a < c ? a : (b < c ? c : b));
  }
};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct InRangeOp : public TernaryOp<LOP, MOP, ROP, LPARM, MPARM, RPARM> {
  __device__ static float exec(int i, const typename ParmTriple<LPARM, MPARM, RPARM>::DeviceParm &p) {
    float a = LOP::exec(i, p.left);
    return (a >= MOP::exec(i, p.middle) && a <= ROP::exec(i, p.right)) ? 1 : 0;
  }
};

template<class LOP, class LPARM>
struct NegateOp : public UnaryOp<LOP, LPARM> {
  __device__ static float exec(int i, const typename LPARM::DeviceParm &p) {
    return -LOP::exec(i,p);
  }
};

template<class LOP, class ROP, class LPARM, class RPARM>
struct MultOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(i,p.left) * ROP::exec(i,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct AddOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(i,p.left) + ROP::exec(i,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct SubOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(i,p.left) - ROP::exec(i,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct DivOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i,  const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(i,p.left) / ROP::exec(i,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct MinOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i,  const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return fminf(LOP::exec(i,p.left), ROP::exec(i,p.right));
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct MaxOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static float exec(int i,  const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return fmaxf(LOP::exec(i, p.left),ROP::exec(i, p.right));
  }
};



template<class PARM>
struct LeafOp {
  __device__ static float exec(int i,  const typename PARM::DeviceParm &p) {
    return p.value(i);
  }
  static bool validate(const Range &rng, const PARM &p) {
    return p.validate(rng);
  } 

};

struct IdentityParm {
  struct DeviceParm {
    __device__ float value(int i) const { return i; }
  };

  IdentityParm() { }
  bool validate(const Range &rng) const { return true; }
  void set_kernel_data(DeviceParm &dp) const {}
};

// Param classes all take the appropriate data in a constructor, and have a method
// called value(int i) that returns the param's value for loop index i.
struct ConstantParm {
  struct DeviceParm {
    float _value;
    __device__ float value(int i) const { return _value; }
  };

  float _value;

  ConstantParm(float f) : _value(f) { }
  bool validate(const Range &rng) const { return true; }
  void set_kernel_data(DeviceParm &dp) const { dp._value = _value; }

};

struct ArrayLookupParm {
  struct DeviceParm {
    const float *_ptr;
    int _size;
    int _shift;

    __device__ float value(int i) const { 
	int idx = i+_shift;
	return _ptr[idx < 0 ? idx + _size : idx >= _size ? idx - _size : idx];
//      return _ptr[((i+_shift)+_size)%_size]; 
    };
  };

  const float *_ptr;
  int _shift;
  Range _rng;

  ArrayLookupParm(const DeviceArray1D &grid, int shift) :
    _ptr(&grid.at(0)), _shift(shift), _rng(grid.range()) { }

  bool  validate(const Range &rng) const { 
    return true;
    //return _rng.contains(rng.shift(_shift));  
  }

  void set_kernel_data(DeviceParm &dp) const { dp._ptr = _ptr; dp._shift = _shift;  dp._size = _rng.maxx+1; }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// operators
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<MultOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
mult(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<MultOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<MultOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
operator*(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return mult(left, right);
};


template<class LOP, class LPARM, class MOP, class MPARM, class ROP, class RPARM>
OpWithParm<InRangeOp<LOP, MOP, ROP, LPARM, MPARM, RPARM>, ParmTriple<LPARM, MPARM, RPARM> >
inrange(const OpWithParm<LOP, LPARM> &left, const OpWithParm<MOP, MPARM> &middle, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<InRangeOp<LOP, MOP, ROP, LPARM, MPARM, RPARM>, ParmTriple<LPARM, MPARM, RPARM> >(
    ParmTriple<LPARM, MPARM, RPARM>(left.parm, middle.parm, right.parm) );
};

template<class LOP, class LPARM, class MOP, class MPARM, class ROP, class RPARM>
OpWithParm<MedianOp<LOP, MOP, ROP, LPARM, MPARM, RPARM>, ParmTriple<LPARM, MPARM, RPARM> >
median(const OpWithParm<LOP, LPARM> &left, const OpWithParm<MOP, MPARM> &middle, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<MedianOp<LOP, MOP, ROP, LPARM, MPARM, RPARM>, ParmTriple<LPARM, MPARM, RPARM> >(
    ParmTriple<LPARM, MPARM, RPARM>(left.parm, middle.parm, right.parm) );
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<MinOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
minop(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<MinOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<MaxOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
maxop(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<MaxOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<AddOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
add(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<AddOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<AddOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
operator+(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return add(left, right); 
};


template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<DivOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
div(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<DivOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};


template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<DivOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
operator/(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return div(left, right); 
};


template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<SubOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
sub(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return OpWithParm<SubOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >(ParmPair<LPARM, RPARM>(left.parm, right.parm));
};

template<class LOP, class LPARM, class ROP, class RPARM>
OpWithParm<SubOp<LOP, ROP, LPARM, RPARM>, ParmPair<LPARM, RPARM> >
operator-(const OpWithParm<LOP, LPARM> &left, const OpWithParm<ROP, RPARM> &right)
{
  return sub(left, right); 
};



OpWithParm<LeafOp<ArrayLookupParm>, ArrayLookupParm>
read(const DeviceArray1D &g, int shift=0) {
  return OpWithParm<LeafOp<ArrayLookupParm>, ArrayLookupParm>(ArrayLookupParm(g, shift));
}

OpWithParm<LeafOp<ArrayLookupParm>, ArrayLookupParm>
DeviceArray1D::operator[](int i) const {
  return read(*this, i); 
}

OpWithParm<LeafOp<ConstantParm>, ConstantParm> 
constant(float f) {
  return OpWithParm<LeafOp<ConstantParm>, ConstantParm>(f);
}

OpWithParm<LeafOp<IdentityParm>, IdentityParm> 
identity() {
  return OpWithParm<LeafOp<IdentityParm>, IdentityParm>(IdentityParm());
}



template <typename T>
__global__ void kernel_assign(T functor, float *dst, int nx)
{
  // transpose for coalescing since k is the fastest changing index 
  int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if (i < nx) {
    dst[i] = functor.exec(i);
  }
}


template <typename OP, typename PARM>
bool execute_expression(const OpWithParm<OP,PARM> &func, DeviceArray1D &dst)
{
  if (!func.validate(dst.range())) {
    printf("Range invalid:\n");
    dst.range().print();
    return false;
  }

  typename OpWithParm<OP,PARM>::DeviceParm dp = func.kernel_data(); 
//printf("passing %d bytes (out of %d)\n", sizeof(dp), sizeof(func));

  int tnx = dst._size;

  int threadsInX = 256;
  int blocksInX = (tnx+threadsInX-1)/threadsInX;
 
  kernel_assign<<<blocksInX, threadsInX>>>(dp, &dst.at(0), dst._size);
  return true; 
}




