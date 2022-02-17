#include "cConvolSpec.h"

template <class tData>
class DigeoConvolution
{
public:
	typedef typename El_CTypeTraits<tData>::tBase tBase;

	DigeoConvolution();

	// return if used kernel is compiled (fast) or not
	bool operator ()( const tData **i_srcData, const int i_width, const int i_height, const Im1D<tBase,tBase> &i_kernel, tData **o_dstData ) const;
	void operator ()( const tData **i_srcData, const int i_width, const int i_height, const cConvolSpec<tData> &i_kernel, tData **o_dstData ) const;

	void getCompiledKernels( vector<vector<tBase> > &o_kernels ) const;

	void generateCode() const;
};

template <class tData>
void LegacyConvolution( const tData *i_src, const int i_width, const int i_height, tData *o_tmp, const vector<typename El_CTypeTraits<tData>::tBase> &i_kernel, tData *o_dst );
