#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num0 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num0(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   5*(INT(In[-5])+INT(In[5]))
				              +   78*(INT(In[-4])+INT(In[4]))
				              +   659*(INT(In[-3])+INT(In[3]))
				              +   3022*(INT(In[-2])+INT(In[2]))
				              +   7523*(INT(In[-1])+INT(In[1]))
				              +   10194*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num0(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15){}
};

#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num1 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num1(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3*(INT(In[-8])+INT(In[8]))
				              +   17*(INT(In[-7])+INT(In[7]))
				              +   84*(INT(In[-6])+INT(In[6]))
				              +   315*(INT(In[-5])+INT(In[5]))
				              +   933*(INT(In[-4])+INT(In[4]))
				              +   2169*(INT(In[-3])+INT(In[3]))
				              +   3963*(INT(In[-2])+INT(In[2]))
				              +   5690*(INT(In[-1])+INT(In[1]))
				              +   6420*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num1(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15){}
};

#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num2 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num2(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3*(INT(In[-10])+INT(In[10]))
				              +   10*(INT(In[-9])+INT(In[9]))
				              +   38*(INT(In[-8])+INT(In[8]))
				              +   120*(INT(In[-7])+INT(In[7]))
				              +   325*(INT(In[-6])+INT(In[6]))
				              +   755*(INT(In[-5])+INT(In[5]))
				              +   1503*(INT(In[-4])+INT(In[4]))
				              +   2569*(INT(In[-3])+INT(In[3]))
				              +   3766*(INT(In[-2])+INT(In[2]))
				              +   4738*(INT(In[-1])+INT(In[1]))
				              +   5114*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num2(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15){}
};

#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num3 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num3(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4*(INT(In[-12])+INT(In[12]))
				              +   12*(INT(In[-11])+INT(In[11]))
				              +   32*(INT(In[-10])+INT(In[10]))
				              +   80*(INT(In[-9])+INT(In[9]))
				              +   183*(INT(In[-8])+INT(In[8]))
				              +   379*(INT(In[-7])+INT(In[7]))
				              +   712*(INT(In[-6])+INT(In[6]))
				              +   1212*(INT(In[-5])+INT(In[5]))
				              +   1875*(INT(In[-4])+INT(In[4]))
				              +   2631*(INT(In[-3])+INT(In[3]))
				              +   3352*(INT(In[-2])+INT(In[2]))
				              +   3877*(INT(In[-1])+INT(In[1]))
				              +   4070*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num3(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15){}
};

#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num4 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num4(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3*(INT(In[-15])+INT(In[15]))
				              +   8*(INT(In[-14])+INT(In[14]))
				              +   18*(INT(In[-13])+INT(In[13]))
				              +   39*(INT(In[-12])+INT(In[12]))
				              +   80*(INT(In[-11])+INT(In[11]))
				              +   152*(INT(In[-10])+INT(In[10]))
				              +   271*(INT(In[-9])+INT(In[9]))
				              +   456*(INT(In[-8])+INT(In[8]))
				              +   722*(INT(In[-7])+INT(In[7]))
				              +   1075*(INT(In[-6])+INT(In[6]))
				              +   1505*(INT(In[-5])+INT(In[5]))
				              +   1982*(INT(In[-4])+INT(In[4]))
				              +   2456*(INT(In[-3])+INT(In[3]))
				              +   2862*(INT(In[-2])+INT(In[2]))
				              +   3137*(INT(In[-1])+INT(In[1]))
				              +   3236*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num4(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-15),-15,15,15){}
};

#define HAS_U_INT2_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT2_Num5 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT2> * duplicate() const { return new cConvolSpec_U_INT2_Num5(*this); }
		void Convol(U_INT2 *Out, const U_INT2 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-19])+INT(In[19]))
				              +   5*(INT(In[-18])+INT(In[18]))
				              +   10*(INT(In[-17])+INT(In[17]))
				              +   18*(INT(In[-16])+INT(In[16]))
				              +   33*(INT(In[-15])+INT(In[15]))
				              +   58*(INT(In[-14])+INT(In[14]))
				              +   98*(INT(In[-13])+INT(In[13]))
				              +   159*(INT(In[-12])+INT(In[12]))
				              +   248*(INT(In[-11])+INT(In[11]))
				              +   372*(INT(In[-10])+INT(In[10]))
				              +   538*(INT(In[-9])+INT(In[9]))
				              +   746*(INT(In[-8])+INT(In[8]))
				              +   997*(INT(In[-7])+INT(In[7]))
				              +   1282*(INT(In[-6])+INT(In[6]))
				              +   1586*(INT(In[-5])+INT(In[5]))
				              +   1887*(INT(In[-4])+INT(In[4]))
				              +   2160*(INT(In[-3])+INT(In[3]))
				              +   2379*(INT(In[-2])+INT(In[2]))
				              +   2521*(INT(In[-1])+INT(In[1]))
				              +   2570*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num5(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-19),-19,19,15){}
};

template <> void ConvolutionHandler<U_INT2>::addCompiledKernels()
{
	{
		INT theCoeff[11] = {5,78,659,3022,7523,10194,7523,3022,659,78,5};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num0(theCoeff) );
	}
	{
		INT theCoeff[17] = {3,17,84,315,933,2169,3963,5690,6420,5690,3963,2169,933,315,84,17,3};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num1(theCoeff) );
	}
	{
		INT theCoeff[21] = {3,10,38,120,325,755,1503,2569,3766,4738,5114,4738,3766,2569,1503,755,325,120,38,10,3};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num2(theCoeff) );
	}
	{
		INT theCoeff[25] = {4,12,32,80,183,379,712,1212,1875,2631,3352,3877,4070,3877,3352,2631,1875,1212,712,379,183,80,32,12,4};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num3(theCoeff) );
	}
	{
		INT theCoeff[31] = {3,8,18,39,80,152,271,456,722,1075,1505,1982,2456,2862,3137,3236,3137,2862,2456,1982,1505,1075,722,456,271,152,80,39,18,8,3};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num4(theCoeff) );
	}
	{
		INT theCoeff[39] = {2,5,10,18,33,58,98,159,248,372,538,746,997,1282,1586,1887,2160,2379,2521,2570,2521,2379,2160,1887,1586,1282,997,746,538,372,248,159,98,58,33,18,10,5,2};
		mConvolutions.push_back( new cConvolSpec_U_INT2_Num5(theCoeff) );
	}
}
