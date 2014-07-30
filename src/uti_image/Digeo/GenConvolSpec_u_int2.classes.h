class cConvolSpec_U_INT2_Num0 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   23*(In[-3]+In[3])
				              +   883*(In[-2]+In[2])
				              +   7662*(In[-1]+In[1])
				              +   15632*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num0(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false){}
};

class cConvolSpec_U_INT2_Num1 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   5*(In[-4]+In[4])
				              +   162*(In[-3]+In[3])
				              +   1852*(In[-2]+In[2])
				              +   7933*(In[-1]+In[1])
				              +   12864*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num1(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false){}
};

class cConvolSpec_U_INT2_Num2 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   4*(In[-5]+In[5])
				              +   67*(In[-4]+In[4])
				              +   609*(In[-3]+In[3])
				              +   2945*(In[-2]+In[2])
				              +   7573*(In[-1]+In[1])
				              +   10372*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num2(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_U_INT2_Num3 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   6*(In[-6]+In[6])
				              +   53*(In[-5]+In[5])
				              +   326*(In[-4]+In[4])
				              +   1346*(In[-3]+In[3])
				              +   3702*(In[-2]+In[2])
				              +   6793*(In[-1]+In[1])
				              +   8316*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num3(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false){}
};

class cConvolSpec_U_INT2_Num4 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   2*(In[-8]+In[8])
				              +   12*(In[-7]+In[7])
				              +   64*(In[-6]+In[6])
				              +   263*(In[-5]+In[5])
				              +   842*(In[-4]+In[4])
				              +   2078*(In[-3]+In[3])
				              +   3964*(In[-2]+In[2])
				              +   5838*(In[-1]+In[1])
				              +   6642*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num4(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false){}
};

class cConvolSpec_U_INT2_Num5 : public cConvolSpec<U_INT2>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  16384
				              +   1*(In[-10]+In[10])
				              +   7*(In[-9]+In[9])
				              +   28*(In[-8]+In[8])
				              +   95*(In[-7]+In[7])
				              +   276*(In[-6]+In[6])
				              +   682*(In[-5]+In[5])
				              +   1426*(In[-4]+In[4])
				              +   2531*(In[-3]+In[3])
				              +   3814*(In[-2]+In[2])
				              +   4877*(In[-1]+In[1])
				              +   5294*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT2_Num5(INT * aFilter):cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false){}
};

