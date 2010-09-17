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

struct EmptyStruct {
};

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
  static bool validate(const Range &ord_rng, const Range &idx_rng, const LPARM &p) {
    return LOP::validate(ord_rng, idx_rng, p);
  } 

};

// Op classes have a method called exec(int i, Param p) that returns the value for loop index i
// and param p.  This might call p.value, or pass p down to their children
template<class LOP, class ROP, class LPARM, class RPARM>
struct BinaryOp {
  static bool validate(const Range &ord_rng, const Range &idx_rng, const ParmPair<LPARM, RPARM> &p) {
    return LOP::validate(ord_rng, idx_rng, p.left) && ROP::validate(ord_rng, idx_rng, p.right);
  } 

};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct TernaryOp {
  static bool validate(const Range &ord_rng, const Range &idx_rng, const ParmTriple<LPARM, MPARM, RPARM> &p) {
    return LOP::validate(ord_rng, idx_rng, p.left) && 
           MOP::validate(ord_rng, idx_rng, p.middle), 
           ROP::validate(ord_rng, idx_rng, p.right);
  } 

};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct MedianOp : public TernaryOp<LOP, MOP, ROP, LPARM, MPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmTriple<LPARM, MPARM, RPARM>::DeviceParm &p) {
    double a = LOP::exec(ordinal, index, p.left);
    double b = MOP::exec(ordinal, index, p.middle);
    double c = ROP::exec(ordinal, index, p.right); 
    return a < b ? 
      (b < c ? b : (a < c ? c : a)) : 
      (a < c ? a : (b < c ? c : b));
  }
};

template<class LOP, class MOP, class ROP, class LPARM, class MPARM, class RPARM>
struct InRangeOp : public TernaryOp<LOP, MOP, ROP, LPARM, MPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmTriple<LPARM, MPARM, RPARM>::DeviceParm &p) {
    double a = LOP::exec(ordinal, index, p.left);
    return (a >= MOP::exec(ordinal, index, p.middle) && a <= ROP::exec(ordinal, index, p.right)) ? 1 : 0;
  }
};

template<class LOP, class LPARM>
struct NegateOp : public UnaryOp<LOP, LPARM> {
  __device__ static double exec(int ordinal, int index, const typename LPARM::DeviceParm &p) {
    return -LOP::exec(ordinal, index, p);
  }
};

template<class LOP, class ROP, class LPARM, class RPARM>
struct MultOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(ordinal, index,p.left) * ROP::exec(ordinal, index,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct AddOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(ordinal,index,p.left) + ROP::exec(ordinal,index,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct SubOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(ordinal,index,p.left) - ROP::exec(ordinal,index,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct DivOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index, const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return LOP::exec(ordinal, index, p.left) / ROP::exec(ordinal,index,p.right); 
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct MinOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index,  const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return fminf(LOP::exec(ordinal,index,p.left), ROP::exec(ordinal, index,p.right));
  }
};


template<class LOP, class ROP, class LPARM, class RPARM>
struct MaxOp : public BinaryOp<LOP, ROP, LPARM, RPARM> {
  __device__ static double exec(int ordinal, int index,  const typename ParmPair<LPARM, RPARM>::DeviceParm &p) {
    return fmaxf(LOP::exec(ordinal, index, p.left),ROP::exec(ordinal, index, p.right));
  }
};



template<class PARM>
struct LeafOp {
  __device__ static double exec(int ordinal, int index,  const typename PARM::DeviceParm &p) {
    return p.value(ordinal, index);
  }
  static bool validate(const Range &ord_rng, const Range &idx_rng, const PARM &p) {
    return p.validate(ord_rng, idx_rng);
  } 

};

struct IndexParm {
  struct DeviceParm {
    __device__ double value(int ordinal, int index) const { return index; }
  };

  IndexParm() { }
  bool validate(const Range &ord_rng, const Range &idx_rng) const { return true; }
  void set_kernel_data(DeviceParm &dp) const {}
};

struct RandParm {
  struct DeviceParm {
    int _seed;
    __device__ double value(int ordinal, int index) const { 
      index ^= _seed;
      index = (index+0x7ed55d16) + (index<<12);
      index = (index^0xc761c23c) ^ (index>>19);
      index = (index+0x165667b1) + (index<<5);
      index = (index+0xd3a2646c) ^ (index<<9);
      index = (index+0xfd7046c5) + (index<<3);
      index = (index^0xb55a4f09) ^ (index>>16);

      return (index + 1.0f) / 4294967296.0f;
    }
  };

  int _seed;
  RandParm(int seed) : _seed(seed) { }
  bool validate(const Range &ord_rng, const Range &idx_rng) const { return true; }
  void set_kernel_data(DeviceParm &dp) const { dp._seed = _seed; }
};


// Param classes all take the appropriate data in a constructor, and have a method
// called value(int i) that returns the param's value for loop index i.
struct ConstantParm {
  struct DeviceParm {
    double _value;
    __device__ double value(int ordinal, int index) const { return _value; }
  };

  double _value;

  ConstantParm(double f) : _value(f) { }
  bool validate(const Range &ord_rng, const Range &idx_rng) const { return true; }
  void set_kernel_data(DeviceParm &dp) const { dp._value = _value; }

};

struct ArrayLookupIndexParm {
  struct DeviceParm {
    const double *_ptr;
    int _shift;

    __device__ double value(int ordinal, int index) const { 
	return _ptr[index + _shift];
    };
  };

  const double *_ptr;
  int _shift;
  Range _rng;  // valid range of _ptr[i] accesses

  ArrayLookupIndexParm(const DeviceArray1D &grid, int shift) :
    _ptr(&grid.at(0)), _shift(shift), _rng(grid.valid_range()) { }

  bool  validate(const Range &ord_rng, const Range &idx_rng) const { 
    return _rng.contains(idx_rng.shift(_shift));  
  }

  void set_kernel_data(DeviceParm &dp) const { 
    dp._ptr = _ptr; 
    dp._shift= _shift;
  }


};


struct ArrayLookupOrdinalParm {
  struct DeviceParm {
    const double *_ptr;
    int _start;
    int _stride;

    __device__ double value(int ordinal, int index) const { 
	return _ptr[(ordinal * _stride) + _start];
    };
  };

  const double *_ptr;
  int _start;
  int _stride;
  Range _rng;  // valid range of _ptr[i] accesses

  ArrayLookupOrdinalParm(const DeviceArray1D &grid, int start, int stride) :
    _ptr(&grid.at(0)), _start(start), _stride(stride), _rng(grid.valid_range()) { }

  bool  validate(const Range &ord_rng, const Range &idx_rng) const { 
    return _rng.contains(ord_rng.stride(_stride).shift(_start));  
  }

  void set_kernel_data(DeviceParm &dp) const { 
    dp._ptr = _ptr; 
    dp._start = _start;  
    dp._stride = _stride;
  }

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


template<class OP, class PARM>
OpWithParm<NegateOp<OP, PARM>, PARM>
negate(const OpWithParm<OP, PARM> &operand)
{
  return OpWithParm<NegateOp<OP, PARM>, PARM>(operand.parm);
};

template<class OP, class PARM>
OpWithParm<NegateOp<OP, PARM>, PARM>
operator-(const OpWithParm<OP, PARM> &operand)
{
  return negate(operand); 
};



OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm>
array_read_absolute(const DeviceArray1D &g, int start, int stride) {
  return OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm>(ArrayLookupOrdinalParm(g, start, stride));
}


OpWithParm<LeafOp<ArrayLookupIndexParm>, ArrayLookupIndexParm>
array_read_relative(const DeviceArray1D &g, int shift) {
  return OpWithParm<LeafOp<ArrayLookupIndexParm>, ArrayLookupIndexParm>(ArrayLookupIndexParm(g, shift));
}


OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm>
DeviceArray1D::read(int minx, int maxx, int stride) const {
  return array_read_absolute(*this, minx, stride); 
}

OpWithParm<LeafOp<ArrayLookupOrdinalParm>, ArrayLookupOrdinalParm>
DeviceArray1D::read(int minx) const {
  return array_read_absolute(*this, minx, 1); 
}


OpWithParm<LeafOp<ArrayLookupIndexParm>, ArrayLookupIndexParm>
DeviceArray1D::operator[](int i) const {
  return array_read_relative(*this, i); 
}

OpWithParm<LeafOp<ConstantParm>, ConstantParm> 
constant(double f) {
  return OpWithParm<LeafOp<ConstantParm>, ConstantParm>(f);
}

OpWithParm<LeafOp<IndexParm>, IndexParm> 
index() {
  return OpWithParm<LeafOp<IndexParm>, IndexParm>(IndexParm());
}

OpWithParm<LeafOp<RandParm>, RandParm> 
rand(int seed) {
  return OpWithParm<LeafOp<RandParm>, RandParm>(RandParm(seed));
}



template <typename T>
__global__ void kernel_assign(T functor, double *dst, int ordinal_size, int idx_start, int idx_step)
{
  // transpose for coalescing since k is the fastest changing index 
  int ord = blockIdx.x * blockDim.x + threadIdx.x; 

  if (ord < ordinal_size) {
    int index = idx_start + ord * idx_step;
    dst[index] = functor.exec(ord, index);
  }
}


template <typename OP, typename PARM>
bool execute_expression(const OpWithParm<OP,PARM> &func, DeviceArray1D &dst, int ordinal_size, int idx_start, int idx_step)
{
  Range ord_range(0, ordinal_size-1);
  Range idx_range(idx_start, idx_start + (ordinal_size-1) * idx_step);

  typename OpWithParm<OP,PARM>::DeviceParm dp = func.kernel_data(); 
//printf("passing %d bytes (out of %d)\n", sizeof(dp), sizeof(func));

  if (!func.validate(ord_range, idx_range)) {
    printf("Range invalid (ord, idx):\n");
    ord_range.print();
    idx_range.print();
    return false;
  }


  int threadsInX = 512;
  int blocksInX = (ordinal_size+threadsInX-1)/threadsInX;
 
  kernel_assign<<<blocksInX, threadsInX>>>(dp, &dst.at(0), ordinal_size, idx_start, idx_step); 
  cudaThreadSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e) {
    printf("Error: %s\n", cudaGetErrorString(e));
  }

  return true; 
}




