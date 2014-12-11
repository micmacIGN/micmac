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

class cConvolSpec_REAL4_Num6 : public cConvolSpec<REAL4>
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
				              +   0.000549822*(In[-3]+In[3])
				              +   0.0245943*(In[-2]+In[2])
				              +   0.231898*(In[-1]+In[1])
				              +   0.485916*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num6(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false){}
};

class cConvolSpec_REAL4_Num7 : public cConvolSpec<REAL4>
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
				              +   0.000142936*(In[-4]+In[4])
				              +   0.00466482*(In[-3]+In[3])
				              +   0.055337*(In[-2]+In[2])
				              +   0.242136*(In[-1]+In[1])
				              +   0.395438*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num7(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false){}
};

class cConvolSpec_REAL4_Num8 : public cConvolSpec<REAL4>
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
				              +   0.000105142*(In[-5]+In[5])
				              +   0.00189909*(In[-4]+In[4])
				              +   0.0179319*(In[-3]+In[3])
				              +   0.0888484*(In[-2]+In[2])
				              +   0.23175*(In[-1]+In[1])
				              +   0.318931*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num8(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num9 : public cConvolSpec<REAL4>
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
				              +   0.000155312*(In[-6]+In[6])
				              +   0.0014968*(In[-5]+In[5])
				              +   0.00953945*(In[-4]+In[4])
				              +   0.0402412*(In[-3]+In[3])
				              +   0.112445*(In[-2]+In[2])
				              +   0.208254*(In[-1]+In[1])
				              +   0.255736*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num9(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false){}
};

class cConvolSpec_REAL4_Num10 : public cConvolSpec<REAL4>
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
				              +   4.60885e-05*(In[-8]+In[8])
				              +   0.000330197*(In[-7]+In[7])
				              +   0.00181842*(In[-6]+In[6])
				              +   0.00769928*(In[-5]+In[5])
				              +   0.0250683*(In[-4]+In[4])
				              +   0.0627755*(In[-3]+In[3])
				              +   0.120923*(In[-2]+In[2])
				              +   0.179193*(In[-1]+In[1])
				              +   0.204294*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num10(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false){}
};

class cConvolSpec_REAL4_Num11 : public cConvolSpec<REAL4>
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

		cConvolSpec_REAL4_Num11(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num12 : public cConvolSpec<REAL4>
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

		cConvolSpec_REAL4_Num12(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false){}
};

class cConvolSpec_REAL4_Num13 : public cConvolSpec<REAL4>
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
				              +   5.65993e-05*(In[-5]+In[5])
				              +   0.00129892*(In[-4]+In[4])
				              +   0.0147498*(In[-3]+In[3])
				              +   0.0832804*(In[-2]+In[2])
				              +   0.234812*(In[-1]+In[1])
				              +   0.331605*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num13(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num14 : public cConvolSpec<REAL4>
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
				              +   0.000109044*(In[-5]+In[5])
				              +   0.00194196*(In[-4]+In[4])
				              +   0.0181382*(In[-3]+In[3])
				              +   0.0891801*(In[-2]+In[2])
				              +   0.231548*(In[-1]+In[1])
				              +   0.318166*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num14(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num15 : public cConvolSpec<REAL4>
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
				              +   0.000105786*(In[-5]+In[5])
				              +   0.00189321*(In[-4]+In[4])
				              +   0.0178475*(In[-3]+In[3])
				              +   0.0886263*(In[-2]+In[2])
				              +   0.231821*(In[-1]+In[1])
				              +   0.319412*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num15(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num16 : public cConvolSpec<REAL4>
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
				              +   7.98401e-05*(In[-5]+In[5])
				              +   0.00159169*(In[-4]+In[4])
				              +   0.0163188*(In[-3]+In[3])
				              +   0.0860415*(In[-2]+In[2])
				              +   0.233303*(In[-1]+In[1])
				              +   0.32533*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num16(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false){}
};

class cConvolSpec_REAL4_Num17 : public cConvolSpec<REAL4>
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
				              +   9.00541e-06*(In[-7]+In[7])
				              +   0.000137119*(In[-6]+In[6])
				              +   0.00137327*(In[-5]+In[5])
				              +   0.00904643*(In[-4]+In[4])
				              +   0.0391976*(In[-3]+In[3])
				              +   0.111713*(In[-2]+In[2])
				              +   0.209416*(In[-1]+In[1])
				              +   0.258214*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num17(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false){}
};

class cConvolSpec_REAL4_Num18 : public cConvolSpec<REAL4>
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
				              +   4.40559e-05*(In[-8]+In[8])
				              +   0.000318865*(In[-7]+In[7])
				              +   0.00177254*(In[-6]+In[6])
				              +   0.00756781*(In[-5]+In[5])
				              +   0.024816*(In[-4]+In[4])
				              +   0.0624999*(In[-3]+In[3])
				              +   0.120896*(In[-2]+In[2])
				              +   0.179611*(In[-1]+In[1])
				              +   0.204946*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num18(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false){}
};

class cConvolSpec_REAL4_Num19 : public cConvolSpec<REAL4>
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
				              +   3.99205e-05*(In[-10]+In[10])
				              +   0.000193694*(In[-9]+In[9])
				              +   0.000795854*(In[-8]+In[8])
				              +   0.00276917*(In[-7]+In[7])
				              +   0.00815948*(In[-6]+In[6])
				              +   0.0203598*(In[-5]+In[5])
				              +   0.0430213*(In[-4]+In[4])
				              +   0.0769822*(In[-3]+In[3])
				              +   0.116653*(In[-2]+In[2])
				              +   0.149692*(In[-1]+In[1])
				              +   0.162667*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num19(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false){}
};

class cConvolSpec_REAL4_Num20 : public cConvolSpec<REAL4>
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
				              +   1.85148e-05*(In[-13]+In[13])
				              +   6.85605e-05*(In[-12]+In[12])
				              +   0.000228636*(In[-11]+In[11])
				              +   0.000686644*(In[-10]+In[10])
				              +   0.00185709*(In[-9]+In[9])
				              +   0.00452326*(In[-8]+In[8])
				              +   0.00992168*(In[-7]+In[7])
				              +   0.019599*(In[-6]+In[6])
				              +   0.0348657*(In[-5]+In[5])
				              +   0.0558571*(In[-4]+In[4])
				              +   0.0805887*(In[-3]+In[3])
				              +   0.104709*(In[-2]+In[2])
				              +   0.122521*(In[-1]+In[1])
				              +   0.129108*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num20(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-13),-13,13,15,false){}
};

