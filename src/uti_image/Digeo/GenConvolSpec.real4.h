#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num0 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num0(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   0.000151904*(REAL(In[-5])+REAL(In[5]))
				              +   0.0023789*(REAL(In[-4])+REAL(In[4]))
				              +   0.0201214*(REAL(In[-3])+REAL(In[3]))
				              +   0.0922127*(REAL(In[-2])+REAL(In[2]))
				              +   0.229585*(REAL(In[-1])+REAL(In[1]))
				              +   0.311101*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num0(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num1 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num1(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   8.69949e-05*(REAL(In[-8])+REAL(In[8]))
				              +   0.000531622*(REAL(In[-7])+REAL(In[7]))
				              +   0.00255109*(REAL(In[-6])+REAL(In[6]))
				              +   0.00961463*(REAL(In[-5])+REAL(In[5]))
				              +   0.0284632*(REAL(In[-4])+REAL(In[4]))
				              +   0.066196*(REAL(In[-3])+REAL(In[3]))
				              +   0.120954*(REAL(In[-2])+REAL(In[2]))
				              +   0.173653*(REAL(In[-1])+REAL(In[1]))
				              +   0.195899*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num1(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num2 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num2(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   7.40384e-05*(REAL(In[-10])+REAL(In[10]))
				              +   0.000317041*(REAL(In[-9])+REAL(In[9]))
				              +   0.00116476*(REAL(In[-8])+REAL(In[8]))
				              +   0.00367142*(REAL(In[-7])+REAL(In[7]))
				              +   0.00992938*(REAL(In[-6])+REAL(In[6]))
				              +   0.0230416*(REAL(In[-5])+REAL(In[5]))
				              +   0.0458795*(REAL(In[-4])+REAL(In[4]))
				              +   0.0783874*(REAL(In[-3])+REAL(In[3]))
				              +   0.114922*(REAL(In[-2])+REAL(In[2]))
				              +   0.144576*(REAL(In[-1])+REAL(In[1]))
				              +   0.156073*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num2(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num3 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num3(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   0.000116104*(REAL(In[-12])+REAL(In[12]))
				              +   0.000353763*(REAL(In[-11])+REAL(In[11]))
				              +   0.000978342*(REAL(In[-10])+REAL(In[10]))
				              +   0.00245576*(REAL(In[-9])+REAL(In[9]))
				              +   0.00559499*(REAL(In[-8])+REAL(In[8]))
				              +   0.01157*(REAL(In[-7])+REAL(In[7]))
				              +   0.0217167*(REAL(In[-6])+REAL(In[6]))
				              +   0.036998*(REAL(In[-5])+REAL(In[5]))
				              +   0.0572124*(REAL(In[-4])+REAL(In[4]))
				              +   0.080303*(REAL(In[-3])+REAL(In[3]))
				              +   0.102307*(REAL(In[-2])+REAL(In[2]))
				              +   0.118306*(REAL(In[-1])+REAL(In[1]))
				              +   0.124177*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num3(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-12),-12,12,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num4 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num4(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   0.000100894*(REAL(In[-15])+REAL(In[15]))
				              +   0.000245085*(REAL(In[-14])+REAL(In[14]))
				              +   0.000559996*(REAL(In[-13])+REAL(In[13]))
				              +   0.00120356*(REAL(In[-12])+REAL(In[12]))
				              +   0.00243312*(REAL(In[-11])+REAL(In[11]))
				              +   0.00462676*(REAL(In[-10])+REAL(In[10]))
				              +   0.00827572*(REAL(In[-9])+REAL(In[9]))
				              +   0.0139236*(REAL(In[-8])+REAL(In[8]))
				              +   0.0220351*(REAL(In[-7])+REAL(In[7]))
				              +   0.0328016*(REAL(In[-6])+REAL(In[6]))
				              +   0.0459296*(REAL(In[-5])+REAL(In[5]))
				              +   0.0604936*(REAL(In[-4])+REAL(In[4]))
				              +   0.0749452*(REAL(In[-3])+REAL(In[3]))
				              +   0.0873368*(REAL(In[-2])+REAL(In[2]))
				              +   0.0957346*(REAL(In[-1])+REAL(In[1]))
				              +   0.0987097*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num4(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-15),-15,15,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num5 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num5(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   7.3478e-05*(REAL(In[-19])+REAL(In[19]))
				              +   0.000150154*(REAL(In[-18])+REAL(In[18]))
				              +   0.000295217*(REAL(In[-17])+REAL(In[17]))
				              +   0.000558426*(REAL(In[-16])+REAL(In[16]))
				              +   0.00101628*(REAL(In[-15])+REAL(In[15]))
				              +   0.00177943*(REAL(In[-14])+REAL(In[14]))
				              +   0.00299759*(REAL(In[-13])+REAL(In[13]))
				              +   0.00485831*(REAL(In[-12])+REAL(In[12]))
				              +   0.00757568*(REAL(In[-11])+REAL(In[11]))
				              +   0.0113653*(REAL(In[-10])+REAL(In[10]))
				              +   0.0164045*(REAL(In[-9])+REAL(In[9]))
				              +   0.0227807*(REAL(In[-8])+REAL(In[8]))
				              +   0.0304364*(REAL(In[-7])+REAL(In[7]))
				              +   0.039124*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483857*(REAL(In[-5])+REAL(In[5]))
				              +   0.0575722*(REAL(In[-4])+REAL(In[4]))
				              +   0.0659071*(REAL(In[-3])+REAL(In[3]))
				              +   0.0725895*(REAL(In[-2])+REAL(In[2]))
				              +   0.07692*(REAL(In[-1])+REAL(In[1]))
				              +   0.0784202*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num5(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-19),-19,19,0){}
};

template <> void ConvolutionHandler<REAL4>::addCompiledKernels()
{
	{
		REAL8 theCoeff[11] = {0.000151904,0.0023789,0.0201214,0.0922127,0.229585,0.311101,0.229585,0.0922127,0.0201214,0.0023789,0.000151904};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num0(theCoeff) );
	}
	{
		REAL8 theCoeff[17] = {8.69949e-05,0.000531622,0.00255109,0.00961463,0.0284632,0.066196,0.120954,0.173653,0.195899,0.173653,0.120954,0.066196,0.0284632,0.00961463,0.00255109,0.000531622,8.69949e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num1(theCoeff) );
	}
	{
		REAL8 theCoeff[21] = {7.40384e-05,0.000317041,0.00116476,0.00367142,0.00992938,0.0230416,0.0458795,0.0783874,0.114922,0.144576,0.156073,0.144576,0.114922,0.0783874,0.0458795,0.0230416,0.00992938,0.00367142,0.00116476,0.000317041,7.40384e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num2(theCoeff) );
	}
	{
		REAL8 theCoeff[25] = {0.000116104,0.000353763,0.000978342,0.00245576,0.00559499,0.01157,0.0217167,0.036998,0.0572124,0.080303,0.102307,0.118306,0.124177,0.118306,0.102307,0.080303,0.0572124,0.036998,0.0217167,0.01157,0.00559499,0.00245576,0.000978342,0.000353763,0.000116104};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num3(theCoeff) );
	}
	{
		REAL8 theCoeff[31] = {0.000100894,0.000245085,0.000559996,0.00120356,0.00243312,0.00462676,0.00827572,0.0139236,0.0220351,0.0328016,0.0459296,0.0604936,0.0749452,0.0873368,0.0957346,0.0987097,0.0957346,0.0873368,0.0749452,0.0604936,0.0459296,0.0328016,0.0220351,0.0139236,0.00827572,0.00462676,0.00243312,0.00120356,0.000559996,0.000245085,0.000100894};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num4(theCoeff) );
	}
	{
		REAL8 theCoeff[39] = {7.3478e-05,0.000150154,0.000295217,0.000558426,0.00101628,0.00177943,0.00299759,0.00485831,0.00757568,0.0113653,0.0164045,0.0227807,0.0304364,0.039124,0.0483857,0.0575722,0.0659071,0.0725895,0.07692,0.0784202,0.07692,0.0725895,0.0659071,0.0575722,0.0483857,0.039124,0.0304364,0.0227807,0.0164045,0.0113653,0.00757568,0.00485831,0.00299759,0.00177943,0.00101628,0.000558426,0.000295217,0.000150154,7.3478e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num5(theCoeff) );
	}
}
