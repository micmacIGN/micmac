class cConvolSpec_REAL4_Num0 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000696849*(In[-3]+In[3])
				              +   0.026943*(In[-2]+In[2])
				              +   0.233812*(In[-1]+In[1])
				              +   0.477097*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num0(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false){}
};

class cConvolSpec_REAL4_Num1 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000159447*(In[-4]+In[4])
				              +   0.00494048*(In[-3]+In[3])
				              +   0.0565189*(In[-2]+In[2])
				              +   0.242087*(In[-1]+In[1])
				              +   0.392589*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num1(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false){}
};

class cConvolSpec_REAL4_Num2 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000117692*(In[-5]+In[5])
				              +   0.0020349*(In[-4]+In[4])
				              +   0.0185777*(In[-3]+In[3])
				              +   0.0898761*(In[-2]+In[2])
				              +   0.231116*(In[-1]+In[1])
				              +   0.316556*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num2(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num3 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000172513*(In[-6]+In[6])
				              +   0.00160608*(In[-5]+In[5])
				              +   0.00995132*(In[-4]+In[4])
				              +   0.0410706*(In[-3]+In[3])
				              +   0.112988*(In[-2]+In[2])
				              +   0.207315*(In[-1]+In[1])
				              +   0.253793*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num3(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false){}
};

class cConvolSpec_REAL4_Num4 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   5.20362e-05*(In[-8]+In[8])
				              +   0.000361674*(In[-7]+In[7])
				              +   0.00194016*(In[-6]+In[6])
				              +   0.00803445*(In[-5]+In[5])
				              +   0.0256894*(In[-4]+In[4])
				              +   0.0634304*(In[-3]+In[3])
				              +   0.120961*(In[-2]+In[2])
				              +   0.178171*(In[-1]+In[1])
				              +   0.20272*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num4(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false){}
};

class cConvolSpec_REAL4_Num5 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   4.4335e-05*(In[-10]+In[10])
				              +   0.000210671*(In[-9]+In[9])
				              +   0.000849478*(In[-8]+In[8])
				              +   0.00290676*(In[-7]+In[7])
				              +   0.00844096*(In[-6]+In[6])
				              +   0.0208027*(In[-5]+In[5])
				              +   0.0435114*(In[-4]+In[4])
				              +   0.0772424*(In[-3]+In[3])
				              +   0.116382*(In[-2]+In[2])
				              +   0.148834*(In[-1]+In[1])
				              +   0.16155*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num5(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false){}
};

