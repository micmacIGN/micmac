class cConvolSpec_U_INT1_Num0 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   9*(In[-6]+In[6])
				              +   71*(In[-5]+In[5])
				              +   390*(In[-4]+In[4])
				              +   1465*(In[-3]+In[3])
				              +   3774*(In[-2]+In[2])
				              +   6655*(In[-1]+In[1])
				              +   8040*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num0(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-6),-6,6,15,false){}
};

class cConvolSpec_U_INT1_Num1 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  -1073741824
				              +   -1*(In[-1])
				              +   0*(In[0])
				              +   1*(In[1])
				            )>>2147483647;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num1(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-1),-1,1,2147483647,false){}
};

