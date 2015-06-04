class cConvolSpec_REAL4_Num0 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000151904*(In[-5]+In[5])
				              +   0.0023789*(In[-4]+In[4])
				              +   0.0201214*(In[-3]+In[3])
				              +   0.0922127*(In[-2]+In[2])
				              +   0.229585*(In[-1]+In[1])
				              +   0.311101*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num0(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num1 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
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

		cConvolSpec_REAL4_Num1(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num2 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000172512*(In[-6]+In[6])
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

		cConvolSpec_REAL4_Num2(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false){}
};

class cConvolSpec_REAL4_Num3 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   5.20361e-05*(In[-8]+In[8])
				              +   0.000361673*(In[-7]+In[7])
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

		cConvolSpec_REAL4_Num3(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false){}
};

class cConvolSpec_REAL4_Num4 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   4.43349e-05*(In[-10]+In[10])
				              +   0.000210671*(In[-9]+In[9])
				              +   0.000849477*(In[-8]+In[8])
				              +   0.00290675*(In[-7]+In[7])
				              +   0.00844096*(In[-6]+In[6])
				              +   0.0208026*(In[-5]+In[5])
				              +   0.0435114*(In[-4]+In[4])
				              +   0.0772424*(In[-3]+In[3])
				              +   0.116382*(In[-2]+In[2])
				              +   0.148834*(In[-1]+In[1])
				              +   0.16155*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num4(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false){}
};

class cConvolSpec_REAL4_Num5 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   7.28214e-05*(In[-12]+In[12])
				              +   0.000240379*(In[-11]+In[11])
				              +   0.000715188*(In[-10]+In[10])
				              +   0.00191793*(In[-9]+In[9])
				              +   0.00463596*(In[-8]+In[8])
				              +   0.0101005*(In[-7]+In[7])
				              +   0.0198356*(In[-6]+In[6])
				              +   0.0351117*(In[-5]+In[5])
				              +   0.0560224*(In[-4]+In[4])
				              +   0.0805711*(In[-3]+In[3])
				              +   0.104449*(In[-2]+In[2])
				              +   0.122051*(In[-1]+In[1])
				              +   0.128554*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num5(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false){}
};

class cConvolSpec_REAL4_Num6 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1)
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				                  0
				              +   0.000269172*(In[-6]+In[6])
				              +   0.0021638*(In[-5]+In[5])
				              +   0.0118935*(In[-4]+In[4])
				              +   0.0447304*(In[-3]+In[3])
				              +   0.115169*(In[-2]+In[2])
				              +   0.203094*(In[-1]+In[1])
				              +   0.24536*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num6(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,0,false){}
};

