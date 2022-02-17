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

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num6 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num6(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.92875e-22*(REAL(In[-1])+REAL(In[1]))
				              +   1*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num6(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-1),-1,1,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num7 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num7(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.21649e-10*(REAL(In[-2])+REAL(In[2]))
				              +   0.00383626*(REAL(In[-1])+REAL(In[1]))
				              +   0.992327*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num7(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-2),-2,2,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num8 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num8(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3.42561e-06*(REAL(In[-2])+REAL(In[2]))
				              +   0.0403876*(REAL(In[-1])+REAL(In[1]))
				              +   0.919218*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num8(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-2),-2,2,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num9 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num9(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   0.000263865*(REAL(In[-2])+REAL(In[2]))
				              +   0.106451*(REAL(In[-1])+REAL(In[1]))
				              +   0.786571*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num9(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-2),-2,2,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num10 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num10(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.47381e-06*(REAL(In[-3])+REAL(In[3]))
				              +   0.00256626*(REAL(In[-2])+REAL(In[2]))
				              +   0.165524*(REAL(In[-1])+REAL(In[1]))
				              +   0.663815*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num10(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-3),-3,3,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num11 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num11(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   5.85246e-05*(REAL(In[-3])+REAL(In[3]))
				              +   0.00961893*(REAL(In[-2])+REAL(In[2]))
				              +   0.2054*(REAL(In[-1])+REAL(In[1]))
				              +   0.569846*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num11(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-3),-3,3,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num12 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num12(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.85839e-06*(REAL(In[-4])+REAL(In[4]))
				              +   0.000440742*(REAL(In[-3])+REAL(In[3]))
				              +   0.0219102*(REAL(In[-2])+REAL(In[2]))
				              +   0.22831*(REAL(In[-1])+REAL(In[1]))
				              +   0.498675*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num12(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-4),-4,4,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num13 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num13(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.27688e-05*(REAL(In[-4])+REAL(In[4]))
				              +   0.00171364*(REAL(In[-3])+REAL(In[3]))
				              +   0.0375263*(REAL(In[-2])+REAL(In[2]))
				              +   0.239103*(REAL(In[-1])+REAL(In[1]))
				              +   0.443269*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num13(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-4),-4,4,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num14 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num14(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   0.000133831*(REAL(In[-4])+REAL(In[4]))
				              +   0.00443186*(REAL(In[-3])+REAL(In[3]))
				              +   0.0539911*(REAL(In[-2])+REAL(In[2]))
				              +   0.241971*(REAL(In[-1])+REAL(In[1]))
				              +   0.398943*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num14(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-4),-4,4,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num15 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num15(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.18305e-05*(REAL(In[-5])+REAL(In[5]))
				              +   0.000487696*(REAL(In[-4])+REAL(In[4]))
				              +   0.00879777*(REAL(In[-3])+REAL(In[3]))
				              +   0.0694505*(REAL(In[-2])+REAL(In[2]))
				              +   0.239915*(REAL(In[-1])+REAL(In[1]))
				              +   0.362675*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num15(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num16 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num16(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   5.64693e-05*(REAL(In[-5])+REAL(In[5]))
				              +   0.00128524*(REAL(In[-4])+REAL(In[4]))
				              +   0.014607*(REAL(In[-3])+REAL(In[3]))
				              +   0.0828978*(REAL(In[-2])+REAL(In[2]))
				              +   0.234927*(REAL(In[-1])+REAL(In[1]))
				              +   0.332453*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num16(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-5),-5,5,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num17 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num17(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   7.26683e-06*(REAL(In[-6])+REAL(In[6]))
				              +   0.000188248*(REAL(In[-5])+REAL(In[5]))
				              +   0.00269858*(REAL(In[-4])+REAL(In[4]))
				              +   0.0214073*(REAL(In[-3])+REAL(In[3]))
				              +   0.0939743*(REAL(In[-2])+REAL(In[2]))
				              +   0.228285*(REAL(In[-1])+REAL(In[1]))
				              +   0.306879*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num17(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num18 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num18(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.92661e-05*(REAL(In[-6])+REAL(In[6]))
				              +   0.000484226*(REAL(In[-5])+REAL(In[5]))
				              +   0.00481008*(REAL(In[-4])+REAL(In[4]))
				              +   0.0286865*(REAL(In[-3])+REAL(In[3]))
				              +   0.102713*(REAL(In[-2])+REAL(In[2]))
				              +   0.220797*(REAL(In[-1])+REAL(In[1]))
				              +   0.284959*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num18(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-6),-6,6,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num19 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num19(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4.96403e-06*(REAL(In[-7])+REAL(In[7]))
				              +   8.92202e-05*(REAL(In[-6])+REAL(In[6]))
				              +   0.00102819*(REAL(In[-5])+REAL(In[5]))
				              +   0.00759733*(REAL(In[-4])+REAL(In[4]))
				              +   0.035994*(REAL(In[-3])+REAL(In[3]))
				              +   0.10934*(REAL(In[-2])+REAL(In[2]))
				              +   0.212965*(REAL(In[-1])+REAL(In[1]))
				              +   0.265962*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num19(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-7),-7,7,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num20 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num20(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.73963e-05*(REAL(In[-7])+REAL(In[7]))
				              +   0.000220373*(REAL(In[-6])+REAL(In[6]))
				              +   0.00188891*(REAL(In[-5])+REAL(In[5]))
				              +   0.0109552*(REAL(In[-4])+REAL(In[4]))
				              +   0.0429915*(REAL(In[-3])+REAL(In[3]))
				              +   0.114156*(REAL(In[-2])+REAL(In[2]))
				              +   0.205101*(REAL(In[-1])+REAL(In[1]))
				              +   0.249339*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num20(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-7),-7,7,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num21 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num21(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4.88348e-05*(REAL(In[-7])+REAL(In[7]))
				              +   0.000462931*(REAL(In[-6])+REAL(In[6]))
				              +   0.00310476*(REAL(In[-5])+REAL(In[5]))
				              +   0.0147321*(REAL(In[-4])+REAL(In[4]))
				              +   0.049457*(REAL(In[-3])+REAL(In[3]))
				              +   0.117467*(REAL(In[-2])+REAL(In[2]))
				              +   0.197391*(REAL(In[-1])+REAL(In[1]))
				              +   0.234674*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num21(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-7),-7,7,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num22 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num22(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.13844e-05*(REAL(In[-8])+REAL(In[8]))
				              +   0.000115245*(REAL(In[-7])+REAL(In[7]))
				              +   0.000856823*(REAL(In[-6])+REAL(In[6]))
				              +   0.00467864*(REAL(In[-5])+REAL(In[5]))
				              +   0.0187632*(REAL(In[-4])+REAL(In[4]))
				              +   0.0552652*(REAL(In[-3])+REAL(In[3]))
				              +   0.119552*(REAL(In[-2])+REAL(In[2]))
				              +   0.18994*(REAL(In[-1])+REAL(In[1]))
				              +   0.221635*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num22(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num23 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num23(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.96796e-05*(REAL(In[-8])+REAL(In[8]))
				              +   0.000236991*(REAL(In[-7])+REAL(In[7]))
				              +   0.0014345*(REAL(In[-6])+REAL(In[6]))
				              +   0.00658217*(REAL(In[-5])+REAL(In[5]))
				              +   0.0228946*(REAL(In[-4])+REAL(In[4]))
				              +   0.0603663*(REAL(In[-3])+REAL(In[3]))
				              +   0.120657*(REAL(In[-2])+REAL(In[2]))
				              +   0.182813*(REAL(In[-1])+REAL(In[1]))
				              +   0.209971*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num23(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-8),-8,8,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num24 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num24(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   7.99188e-06*(REAL(In[-9])+REAL(In[9]))
				              +   6.69152e-05*(REAL(In[-8])+REAL(In[8]))
				              +   0.000436342*(REAL(In[-7])+REAL(In[7]))
				              +   0.00221593*(REAL(In[-6])+REAL(In[6]))
				              +   0.00876416*(REAL(In[-5])+REAL(In[5]))
				              +   0.0269955*(REAL(In[-4])+REAL(In[4]))
				              +   0.0647589*(REAL(In[-3])+REAL(In[3]))
				              +   0.120986*(REAL(In[-2])+REAL(In[2]))
				              +   0.176033*(REAL(In[-1])+REAL(In[1]))
				              +   0.199471*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num24(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-9),-9,9,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num25 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num25(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.95108e-05*(REAL(In[-9])+REAL(In[9]))
				              +   0.000134076*(REAL(In[-8])+REAL(In[8]))
				              +   0.000734422*(REAL(In[-7])+REAL(In[7]))
				              +   0.00320673*(REAL(In[-6])+REAL(In[6]))
				              +   0.0111609*(REAL(In[-5])+REAL(In[5]))
				              +   0.030964*(REAL(In[-4])+REAL(In[4]))
				              +   0.0684755*(REAL(In[-3])+REAL(In[3]))
				              +   0.120707*(REAL(In[-2])+REAL(In[2]))
				              +   0.169611*(REAL(In[-1])+REAL(In[1]))
				              +   0.189973*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num25(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-9),-9,9,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num26 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num26(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4.21131e-05*(REAL(In[-9])+REAL(In[9]))
				              +   0.000243851*(REAL(In[-8])+REAL(In[8]))
				              +   0.00114842*(REAL(In[-7])+REAL(In[7]))
				              +   0.00439894*(REAL(In[-6])+REAL(In[6]))
				              +   0.0137046*(REAL(In[-5])+REAL(In[5]))
				              +   0.0347257*(REAL(In[-4])+REAL(In[4]))
				              +   0.071566*(REAL(In[-3])+REAL(In[3]))
				              +   0.119959*(REAL(In[-2])+REAL(In[2]))
				              +   0.163542*(REAL(In[-1])+REAL(In[1]))
				              +   0.18134*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num26(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-9),-9,9,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num27 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num27(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.36245e-05*(REAL(In[-10])+REAL(In[10]))
				              +   8.20815e-05*(REAL(In[-9])+REAL(In[9]))
				              +   0.000409328*(REAL(In[-8])+REAL(In[8]))
				              +   0.00168967*(REAL(In[-7])+REAL(In[7]))
				              +   0.00577342*(REAL(In[-6])+REAL(In[6]))
				              +   0.0163293*(REAL(In[-5])+REAL(In[5]))
				              +   0.0382302*(REAL(In[-4])+REAL(In[4]))
				              +   0.0740878*(REAL(In[-3])+REAL(In[3]))
				              +   0.118847*(REAL(In[-2])+REAL(In[2]))
				              +   0.15781*(REAL(In[-1])+REAL(In[1]))
				              +   0.173454*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num27(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num28 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num28(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.82349e-05*(REAL(In[-10])+REAL(In[10]))
				              +   0.000146916*(REAL(In[-9])+REAL(In[9]))
				              +   0.000642623*(REAL(In[-8])+REAL(In[8]))
				              +   0.00236289*(REAL(In[-7])+REAL(In[7]))
				              +   0.00730354*(REAL(In[-6])+REAL(In[6]))
				              +   0.0189768*(REAL(In[-5])+REAL(In[5]))
				              +   0.0414492*(REAL(In[-4])+REAL(In[4]))
				              +   0.0761046*(REAL(In[-3])+REAL(In[3]))
				              +   0.117465*(REAL(In[-2])+REAL(In[2]))
				              +   0.152407*(REAL(In[-1])+REAL(In[1]))
				              +   0.166228*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num28(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-10),-10,10,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num29 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num29(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   9.97702e-06*(REAL(In[-11])+REAL(In[11]))
				              +   5.35323e-05*(REAL(In[-10])+REAL(In[10]))
				              +   0.000244762*(REAL(In[-9])+REAL(In[9]))
				              +   0.000953639*(REAL(In[-8])+REAL(In[8]))
				              +   0.00316619*(REAL(In[-7])+REAL(In[7]))
				              +   0.00895784*(REAL(In[-6])+REAL(In[6]))
				              +   0.0215965*(REAL(In[-5])+REAL(In[5]))
				              +   0.0443685*(REAL(In[-4])+REAL(In[4]))
				              +   0.0776747*(REAL(In[-3])+REAL(In[3]))
				              +   0.115877*(REAL(In[-2])+REAL(In[2]))
				              +   0.147309*(REAL(In[-1])+REAL(In[1]))
				              +   0.159577*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num29(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-11),-11,11,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num30 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num30(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.99128e-05*(REAL(In[-11])+REAL(In[11]))
				              +   9.41246e-05*(REAL(In[-10])+REAL(In[10]))
				              +   0.000383732*(REAL(In[-9])+REAL(In[9]))
				              +   0.0013493*(REAL(In[-8])+REAL(In[8]))
				              +   0.00409208*(REAL(In[-7])+REAL(In[7]))
				              +   0.0107037*(REAL(In[-6])+REAL(In[6]))
				              +   0.024148*(REAL(In[-5])+REAL(In[5]))
				              +   0.0469875*(REAL(In[-4])+REAL(In[4]))
				              +   0.0788568*(REAL(In[-3])+REAL(In[3]))
				              +   0.114143*(REAL(In[-2])+REAL(In[2]))
				              +   0.142501*(REAL(In[-1])+REAL(In[1]))
				              +   0.153441*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num30(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-11),-11,11,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num31 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num31(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3.67559e-05*(REAL(In[-11])+REAL(In[11]))
				              +   0.000155187*(REAL(In[-10])+REAL(In[10]))
				              +   0.000571225*(REAL(In[-9])+REAL(In[9]))
				              +   0.0018331*(REAL(In[-8])+REAL(In[8]))
				              +   0.00512851*(REAL(In[-7])+REAL(In[7]))
				              +   0.012509*(REAL(In[-6])+REAL(In[6]))
				              +   0.0265999*(REAL(In[-5])+REAL(In[5]))
				              +   0.0493131*(REAL(In[-4])+REAL(In[4]))
				              +   0.0797024*(REAL(In[-3])+REAL(In[3]))
				              +   0.112307*(REAL(In[-2])+REAL(In[2]))
				              +   0.137964*(REAL(In[-1])+REAL(In[1]))
				              +   0.147759*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num31(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-11),-11,11,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num32 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num32(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.46331e-05*(REAL(In[-12])+REAL(In[12]))
				              +   6.34418e-05*(REAL(In[-11])+REAL(In[11]))
				              +   0.000242114*(REAL(In[-10])+REAL(In[10]))
				              +   0.000813335*(REAL(In[-9])+REAL(In[9]))
				              +   0.00240505*(REAL(In[-8])+REAL(In[8]))
				              +   0.00626015*(REAL(In[-7])+REAL(In[7]))
				              +   0.0143433*(REAL(In[-6])+REAL(In[6]))
				              +   0.0289282*(REAL(In[-5])+REAL(In[5]))
				              +   0.0513567*(REAL(In[-4])+REAL(In[4]))
				              +   0.0802563*(REAL(In[-3])+REAL(In[3]))
				              +   0.110399*(REAL(In[-2])+REAL(In[2]))
				              +   0.133677*(REAL(In[-1])+REAL(In[1]))
				              +   0.14248*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num32(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-12),-12,12,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num33 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num33(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.63282e-05*(REAL(In[-12])+REAL(In[12]))
				              +   0.000103344*(REAL(In[-11])+REAL(In[11]))
				              +   0.000360169*(REAL(In[-10])+REAL(In[10]))
				              +   0.00111452*(REAL(In[-9])+REAL(In[9]))
				              +   0.00306218*(REAL(In[-8])+REAL(In[8]))
				              +   0.0074702*(REAL(In[-7])+REAL(In[7]))
				              +   0.0161806*(REAL(In[-6])+REAL(In[6]))
				              +   0.0311183*(REAL(In[-5])+REAL(In[5]))
				              +   0.0531369*(REAL(In[-4])+REAL(In[4]))
				              +   0.0805633*(REAL(In[-3])+REAL(In[3]))
				              +   0.108452*(REAL(In[-2])+REAL(In[2]))
				              +   0.129628*(REAL(In[-1])+REAL(In[1]))
				              +   0.137568*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num33(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-12),-12,12,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num34 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num34(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.11237e-05*(REAL(In[-13])+REAL(In[13]))
				              +   4.46104e-05*(REAL(In[-12])+REAL(In[12]))
				              +   0.000160091*(REAL(In[-11])+REAL(In[11]))
				              +   0.000514096*(REAL(In[-10])+REAL(In[10]))
				              +   0.00147729*(REAL(In[-9])+REAL(In[9]))
				              +   0.00379869*(REAL(In[-8])+REAL(In[8]))
				              +   0.00874068*(REAL(In[-7])+REAL(In[7]))
				              +   0.0179971*(REAL(In[-6])+REAL(In[6]))
				              +   0.0331593*(REAL(In[-5])+REAL(In[5]))
				              +   0.0546704*(REAL(In[-4])+REAL(In[4]))
				              +   0.0806574*(REAL(In[-3])+REAL(In[3]))
				              +   0.106483*(REAL(In[-2])+REAL(In[2]))
				              +   0.125795*(REAL(In[-1])+REAL(In[1]))
				              +   0.132982*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num34(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-13),-13,13,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num35 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num35(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.9536e-05*(REAL(In[-13])+REAL(In[13]))
				              +   7.17356e-05*(REAL(In[-12])+REAL(In[12]))
				              +   0.000237379*(REAL(In[-11])+REAL(In[11]))
				              +   0.000707876*(REAL(In[-10])+REAL(In[10]))
				              +   0.00190231*(REAL(In[-9])+REAL(In[9]))
				              +   0.00460693*(REAL(In[-8])+REAL(In[8]))
				              +   0.0100543*(REAL(In[-7])+REAL(In[7]))
				              +   0.0197742*(REAL(In[-6])+REAL(In[6]))
				              +   0.0350473*(REAL(In[-5])+REAL(In[5]))
				              +   0.0559781*(REAL(In[-4])+REAL(In[4]))
				              +   0.0805731*(REAL(In[-3])+REAL(In[3]))
				              +   0.104513*(REAL(In[-2])+REAL(In[2]))
				              +   0.122168*(REAL(In[-1])+REAL(In[1]))
				              +   0.128693*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num35(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-13),-13,13,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num36 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num36(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3.25082e-05*(REAL(In[-13])+REAL(In[13]))
				              +   0.000110189*(REAL(In[-12])+REAL(In[12]))
				              +   0.000338743*(REAL(In[-11])+REAL(In[11]))
				              +   0.000944477*(REAL(In[-10])+REAL(In[10]))
				              +   0.00238837*(REAL(In[-9])+REAL(In[9]))
				              +   0.00547772*(REAL(In[-8])+REAL(In[8]))
				              +   0.0113943*(REAL(In[-7])+REAL(In[7]))
				              +   0.0214962*(REAL(In[-6])+REAL(In[6]))
				              +   0.0367812*(REAL(In[-5])+REAL(In[5]))
				              +   0.0570791*(REAL(In[-4])+REAL(In[4]))
				              +   0.0803374*(REAL(In[-3])+REAL(In[3]))
				              +   0.102553*(REAL(In[-2])+REAL(In[2]))
				              +   0.118731*(REAL(In[-1])+REAL(In[1]))
				              +   0.124672*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num36(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-13),-13,13,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num37 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num37(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.49331e-05*(REAL(In[-14])+REAL(In[14]))
				              +   5.15859e-05*(REAL(In[-13])+REAL(In[13]))
				              +   0.000162567*(REAL(In[-12])+REAL(In[12]))
				              +   0.000467362*(REAL(In[-11])+REAL(In[11]))
				              +   0.00122573*(REAL(In[-10])+REAL(In[10]))
				              +   0.00293262*(REAL(In[-9])+REAL(In[9]))
				              +   0.00640084*(REAL(In[-8])+REAL(In[8]))
				              +   0.0127449*(REAL(In[-7])+REAL(In[7]))
				              +   0.0231504*(REAL(In[-6])+REAL(In[6]))
				              +   0.0383618*(REAL(In[-5])+REAL(In[5]))
				              +   0.0579909*(REAL(In[-4])+REAL(In[4]))
				              +   0.0799724*(REAL(In[-3])+REAL(In[3]))
				              +   0.10061*(REAL(In[-2])+REAL(In[2]))
				              +   0.115468*(REAL(In[-1])+REAL(In[1]))
				              +   0.120893*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num37(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-14),-14,14,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num38 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num38(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.44177e-05*(REAL(In[-14])+REAL(In[14]))
				              +   7.85022e-05*(REAL(In[-13])+REAL(In[13]))
				              +   0.000231468*(REAL(In[-12])+REAL(In[12]))
				              +   0.000625938*(REAL(In[-11])+REAL(In[11]))
				              +   0.0015524*(REAL(In[-10])+REAL(In[10]))
				              +   0.00353107*(REAL(In[-9])+REAL(In[9]))
				              +   0.00736614*(REAL(In[-8])+REAL(In[8]))
				              +   0.0140931*(REAL(In[-7])+REAL(In[7]))
				              +   0.0247288*(REAL(In[-6])+REAL(In[6]))
				              +   0.0397952*(REAL(In[-5])+REAL(In[5]))
				              +   0.058734*(REAL(In[-4])+REAL(In[4]))
				              +   0.0795024*(REAL(In[-3])+REAL(In[3]))
				              +   0.0986965*(REAL(In[-2])+REAL(In[2]))
				              +   0.112371*(REAL(In[-1])+REAL(In[1]))
				              +   0.117338*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num38(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-14),-14,14,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num39 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num39(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.17065e-05*(REAL(In[-15])+REAL(In[15]))
				              +   3.82375e-05*(REAL(In[-14])+REAL(In[14]))
				              +   0.000115107*(REAL(In[-13])+REAL(In[13]))
				              +   0.000319343*(REAL(In[-12])+REAL(In[12]))
				              +   0.000816513*(REAL(In[-11])+REAL(In[11]))
				              +   0.00192405*(REAL(In[-10])+REAL(In[10]))
				              +   0.00417845*(REAL(In[-9])+REAL(In[9]))
				              +   0.008363*(REAL(In[-8])+REAL(In[8]))
				              +   0.0154261*(REAL(In[-7])+REAL(In[7]))
				              +   0.026224*(REAL(In[-6])+REAL(In[6]))
				              +   0.0410855*(REAL(In[-5])+REAL(In[5]))
				              +   0.0593233*(REAL(In[-4])+REAL(In[4]))
				              +   0.0789422*(REAL(In[-3])+REAL(In[3]))
				              +   0.0968146*(REAL(In[-2])+REAL(In[2]))
				              +   0.109426*(REAL(In[-1])+REAL(In[1]))
				              +   0.113985*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num39(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-15),-15,15,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num40 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num40(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.88234e-05*(REAL(In[-15])+REAL(In[15]))
				              +   5.76232e-05*(REAL(In[-14])+REAL(In[14]))
				              +   0.000163301*(REAL(In[-13])+REAL(In[13]))
				              +   0.000428418*(REAL(In[-12])+REAL(In[12]))
				              +   0.00104049*(REAL(In[-11])+REAL(In[11]))
				              +   0.00233935*(REAL(In[-10])+REAL(In[10]))
				              +   0.00486905*(REAL(In[-9])+REAL(In[9]))
				              +   0.00938172*(REAL(In[-8])+REAL(In[8]))
				              +   0.0167344*(REAL(In[-7])+REAL(In[7]))
				              +   0.027633*(REAL(In[-6])+REAL(In[6]))
				              +   0.042241*(REAL(In[-5])+REAL(In[5]))
				              +   0.0597766*(REAL(In[-4])+REAL(In[4]))
				              +   0.0783101*(REAL(In[-3])+REAL(In[3]))
				              +   0.0949716*(REAL(In[-2])+REAL(In[2]))
				              +   0.106625*(REAL(In[-1])+REAL(In[1]))
				              +   0.110819*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num40(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-15),-15,15,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num41 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num41(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.90956e-05*(REAL(In[-15])+REAL(In[15]))
				              +   8.39109e-05*(REAL(In[-14])+REAL(In[14]))
				              +   0.00022495*(REAL(In[-13])+REAL(In[13]))
				              +   0.000560569*(REAL(In[-12])+REAL(In[12]))
				              +   0.00129852*(REAL(In[-11])+REAL(In[11]))
				              +   0.00279605*(REAL(In[-10])+REAL(In[10]))
				              +   0.00559653*(REAL(In[-9])+REAL(In[9]))
				              +   0.0104128*(REAL(In[-8])+REAL(In[8]))
				              +   0.0180092*(REAL(In[-7])+REAL(In[7]))
				              +   0.0289532*(REAL(In[-6])+REAL(In[6]))
				              +   0.0432689*(REAL(In[-5])+REAL(In[5]))
				              +   0.060108*(REAL(In[-4])+REAL(In[4]))
				              +   0.0776183*(REAL(In[-3])+REAL(In[3]))
				              +   0.0931693*(REAL(In[-2])+REAL(In[2]))
				              +   0.103958*(REAL(In[-1])+REAL(In[1]))
				              +   0.107825*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num41(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-15),-15,15,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num42 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num42(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.48399e-05*(REAL(In[-16])+REAL(In[16]))
				              +   4.34116e-05*(REAL(In[-15])+REAL(In[15]))
				              +   0.000118496*(REAL(In[-14])+REAL(In[14]))
				              +   0.000301806*(REAL(In[-13])+REAL(In[13]))
				              +   0.000717257*(REAL(In[-12])+REAL(In[12]))
				              +   0.00159055*(REAL(In[-11])+REAL(In[11]))
				              +   0.00329111*(REAL(In[-10])+REAL(In[10]))
				              +   0.00635422*(REAL(In[-9])+REAL(In[9]))
				              +   0.0114474*(REAL(In[-8])+REAL(In[8]))
				              +   0.0192431*(REAL(In[-7])+REAL(In[7]))
				              +   0.0301834*(REAL(In[-6])+REAL(In[6]))
				              +   0.0441758*(REAL(In[-5])+REAL(In[5]))
				              +   0.060329*(REAL(In[-4])+REAL(In[4]))
				              +   0.0768761*(REAL(In[-3])+REAL(In[3]))
				              +   0.0914073*(REAL(In[-2])+REAL(In[2]))
				              +   0.101413*(REAL(In[-1])+REAL(In[1]))
				              +   0.104986*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num42(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-16),-16,16,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num43 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num43(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.26487e-05*(REAL(In[-16])+REAL(In[16]))
				              +   6.27506e-05*(REAL(In[-15])+REAL(In[15]))
				              +   0.000162794*(REAL(In[-14])+REAL(In[14]))
				              +   0.000395465*(REAL(In[-13])+REAL(In[13]))
				              +   0.000899546*(REAL(In[-12])+REAL(In[12]))
				              +   0.00191595*(REAL(In[-11])+REAL(In[11]))
				              +   0.00382115*(REAL(In[-10])+REAL(In[10]))
				              +   0.00713591*(REAL(In[-9])+REAL(In[9]))
				              +   0.0124782*(REAL(In[-8])+REAL(In[8]))
				              +   0.0204315*(REAL(In[-7])+REAL(In[7]))
				              +   0.0313254*(REAL(In[-6])+REAL(In[6]))
				              +   0.0449718*(REAL(In[-5])+REAL(In[5]))
				              +   0.0604546*(REAL(In[-4])+REAL(In[4]))
				              +   0.0760967*(REAL(In[-3])+REAL(In[3]))
				              +   0.089691*(REAL(In[-2])+REAL(In[2]))
				              +   0.0989871*(REAL(In[-1])+REAL(In[1]))
				              +   0.102295*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num43(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-16),-16,16,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num44 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num44(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.19298e-05*(REAL(In[-17])+REAL(In[17]))
				              +   3.34579e-05*(REAL(In[-16])+REAL(In[16]))
				              +   8.81499e-05*(REAL(In[-15])+REAL(In[15]))
				              +   0.000218173*(REAL(In[-14])+REAL(In[14]))
				              +   0.000507268*(REAL(In[-13])+REAL(In[13]))
				              +   0.00110797*(REAL(In[-12])+REAL(In[12]))
				              +   0.00227342*(REAL(In[-11])+REAL(In[11]))
				              +   0.00438213*(REAL(In[-10])+REAL(In[10]))
				              +   0.007935*(REAL(In[-9])+REAL(In[9]))
				              +   0.0134979*(REAL(In[-8])+REAL(In[8]))
				              +   0.0215696*(REAL(In[-7])+REAL(In[7]))
				              +   0.0323798*(REAL(In[-6])+REAL(In[6]))
				              +   0.0456628*(REAL(In[-5])+REAL(In[5]))
				              +   0.0604934*(REAL(In[-4])+REAL(In[4]))
				              +   0.0752852*(REAL(In[-3])+REAL(In[3]))
				              +   0.0880173*(REAL(In[-2])+REAL(In[2]))
				              +   0.0966681*(REAL(In[-1])+REAL(In[1]))
				              +   0.0997367*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num44(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-17),-17,17,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num45 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num45(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.79849e-05*(REAL(In[-17])+REAL(In[17]))
				              +   4.79946e-05*(REAL(In[-16])+REAL(In[16]))
				              +   0.000120682*(REAL(In[-15])+REAL(In[15]))
				              +   0.000285928*(REAL(In[-14])+REAL(In[14]))
				              +   0.000638316*(REAL(In[-13])+REAL(In[13]))
				              +   0.0013427*(REAL(In[-12])+REAL(In[12]))
				              +   0.00266126*(REAL(In[-11])+REAL(In[11]))
				              +   0.00497004*(REAL(In[-10])+REAL(In[10]))
				              +   0.00874575*(REAL(In[-9])+REAL(In[9]))
				              +   0.014501*(REAL(In[-8])+REAL(In[8]))
				              +   0.022655*(REAL(In[-7])+REAL(In[7]))
				              +   0.03335*(REAL(In[-6])+REAL(In[6]))
				              +   0.0462584*(REAL(In[-5])+REAL(In[5]))
				              +   0.0604575*(REAL(In[-4])+REAL(In[4]))
				              +   0.0744517*(REAL(In[-3])+REAL(In[3]))
				              +   0.08639*(REAL(In[-2])+REAL(In[2]))
				              +   0.0944532*(REAL(In[-1])+REAL(In[1]))
				              +   0.0973048*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num45(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-17),-17,17,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num46 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num46(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.63089e-05*(REAL(In[-17])+REAL(In[17]))
				              +   6.70395e-05*(REAL(In[-16])+REAL(In[16]))
				              +   0.000161413*(REAL(In[-15])+REAL(In[15]))
				              +   0.00036722*(REAL(In[-14])+REAL(In[14]))
				              +   0.000789396*(REAL(In[-13])+REAL(In[13]))
				              +   0.0016034*(REAL(In[-12])+REAL(In[12]))
				              +   0.00307731*(REAL(In[-11])+REAL(In[11]))
				              +   0.00558059*(REAL(In[-10])+REAL(In[10]))
				              +   0.00956245*(REAL(In[-9])+REAL(In[9]))
				              +   0.0154824*(REAL(In[-8])+REAL(In[8]))
				              +   0.0236857*(REAL(In[-7])+REAL(In[7]))
				              +   0.0342386*(REAL(In[-6])+REAL(In[6]))
				              +   0.0467655*(REAL(In[-5])+REAL(In[5]))
				              +   0.0603552*(REAL(In[-4])+REAL(In[4]))
				              +   0.0736011*(REAL(In[-3])+REAL(In[3]))
				              +   0.0848074*(REAL(In[-2])+REAL(In[2]))
				              +   0.0923344*(REAL(In[-1])+REAL(In[1]))
				              +   0.0949891*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num46(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-17),-17,17,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num47 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num47(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.45339e-05*(REAL(In[-18])+REAL(In[18]))
				              +   3.74474e-05*(REAL(In[-17])+REAL(In[17]))
				              +   9.14061e-05*(REAL(In[-16])+REAL(In[16]))
				              +   0.000211369*(REAL(In[-15])+REAL(In[15]))
				              +   0.000463039*(REAL(In[-14])+REAL(In[14]))
				              +   0.000960962*(REAL(In[-13])+REAL(In[13]))
				              +   0.00188933*(REAL(In[-12])+REAL(In[12]))
				              +   0.00351901*(REAL(In[-11])+REAL(In[11]))
				              +   0.00620933*(REAL(In[-10])+REAL(In[10]))
				              +   0.0103796*(REAL(In[-9])+REAL(In[9]))
				              +   0.0164373*(REAL(In[-8])+REAL(In[8]))
				              +   0.0246599*(REAL(In[-7])+REAL(In[7]))
				              +   0.035048*(REAL(In[-6])+REAL(In[6]))
				              +   0.0471898*(REAL(In[-5])+REAL(In[5]))
				              +   0.0601927*(REAL(In[-4])+REAL(In[4]))
				              +   0.0727365*(REAL(In[-3])+REAL(In[3]))
				              +   0.0832669*(REAL(In[-2])+REAL(In[2]))
				              +   0.0903035*(REAL(In[-1])+REAL(In[1]))
				              +   0.0927788*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num47(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-18),-18,18,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num48 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num48(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.10568e-05*(REAL(In[-18])+REAL(In[18]))
				              +   5.19951e-05*(REAL(In[-17])+REAL(In[17]))
				              +   0.000121927*(REAL(In[-16])+REAL(In[16]))
				              +   0.000271522*(REAL(In[-15])+REAL(In[15]))
				              +   0.000574218*(REAL(In[-14])+REAL(In[14]))
				              +   0.00115323*(REAL(In[-13])+REAL(In[13]))
				              +   0.0021995*(REAL(In[-12])+REAL(In[12]))
				              +   0.0039838*(REAL(In[-11])+REAL(In[11]))
				              +   0.00685236*(REAL(In[-10])+REAL(In[10]))
				              +   0.0111931*(REAL(In[-9])+REAL(In[9]))
				              +   0.0173631*(REAL(In[-8])+REAL(In[8]))
				              +   0.0255782*(REAL(In[-7])+REAL(In[7]))
				              +   0.0357834*(REAL(In[-6])+REAL(In[6]))
				              +   0.0475402*(REAL(In[-5])+REAL(In[5]))
				              +   0.0599802*(REAL(In[-4])+REAL(In[4]))
				              +   0.0718657*(REAL(In[-3])+REAL(In[3]))
				              +   0.0817717*(REAL(In[-2])+REAL(In[2]))
				              +   0.0883593*(REAL(In[-1])+REAL(In[1]))
				              +   0.090671*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num48(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-18),-18,18,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num49 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num49(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.97412e-05*(REAL(In[-18])+REAL(In[18]))
				              +   7.05788e-05*(REAL(In[-17])+REAL(In[17]))
				              +   0.00015942*(REAL(In[-16])+REAL(In[16]))
				              +   0.000342742*(REAL(In[-15])+REAL(In[15]))
				              +   0.000701364*(REAL(In[-14])+REAL(In[14]))
				              +   0.00136607*(REAL(In[-13])+REAL(In[13]))
				              +   0.00253254*(REAL(In[-12])+REAL(In[12]))
				              +   0.00446881*(REAL(In[-11])+REAL(In[11]))
				              +   0.00750554*(REAL(In[-10])+REAL(In[10]))
				              +   0.0119984*(REAL(In[-9])+REAL(In[9]))
				              +   0.0182567*(REAL(In[-8])+REAL(In[8]))
				              +   0.0264406*(REAL(In[-7])+REAL(In[7]))
				              +   0.0364481*(REAL(In[-6])+REAL(In[6]))
				              +   0.0478224*(REAL(In[-5])+REAL(In[5]))
				              +   0.0597229*(REAL(In[-4])+REAL(In[4]))
				              +   0.0709911*(REAL(In[-3])+REAL(In[3]))
				              +   0.0803195*(REAL(In[-2])+REAL(In[2]))
				              +   0.0864949*(REAL(In[-1])+REAL(In[1]))
				              +   0.0886572*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num49(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-18),-18,18,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num50 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num50(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.71208e-05*(REAL(In[-19])+REAL(In[19]))
				              +   4.10415e-05*(REAL(In[-18])+REAL(In[18]))
				              +   9.38422e-05*(REAL(In[-17])+REAL(In[17]))
				              +   0.000204668*(REAL(In[-16])+REAL(In[16]))
				              +   0.000425771*(REAL(In[-15])+REAL(In[15]))
				              +   0.000844849*(REAL(In[-14])+REAL(In[14]))
				              +   0.00159903*(REAL(In[-13])+REAL(In[13]))
				              +   0.00288676*(REAL(In[-12])+REAL(In[12]))
				              +   0.00497096*(REAL(In[-11])+REAL(In[11]))
				              +   0.00816481*(REAL(In[-10])+REAL(In[10]))
				              +   0.0127917*(REAL(In[-9])+REAL(In[9]))
				              +   0.0191154*(REAL(In[-8])+REAL(In[8]))
				              +   0.0272468*(REAL(In[-7])+REAL(In[7]))
				              +   0.0370445*(REAL(In[-6])+REAL(In[6]))
				              +   0.0480405*(REAL(In[-5])+REAL(In[5]))
				              +   0.0594247*(REAL(In[-4])+REAL(In[4]))
				              +   0.0701136*(REAL(In[-3])+REAL(In[3]))
				              +   0.0789065*(REAL(In[-2])+REAL(In[2]))
				              +   0.0847031*(REAL(In[-1])+REAL(In[1]))
				              +   0.0867285*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num50(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-19),-19,19,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num51 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num51(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.39969e-05*(REAL(In[-19])+REAL(In[19]))
				              +   5.54458e-05*(REAL(In[-18])+REAL(In[18]))
				              +   0.00012244*(REAL(In[-17])+REAL(In[17]))
				              +   0.000258414*(REAL(In[-16])+REAL(In[16]))
				              +   0.000521255*(REAL(In[-15])+REAL(In[15]))
				              +   0.0010049*(REAL(In[-14])+REAL(In[14]))
				              +   0.00185156*(REAL(In[-13])+REAL(In[13]))
				              +   0.00326054*(REAL(In[-12])+REAL(In[12]))
				              +   0.0054876*(REAL(In[-11])+REAL(In[11]))
				              +   0.00882703*(REAL(In[-10])+REAL(In[10]))
				              +   0.0135702*(REAL(In[-9])+REAL(In[9]))
				              +   0.0199388*(REAL(In[-8])+REAL(In[8]))
				              +   0.0279995*(REAL(In[-7])+REAL(In[7]))
				              +   0.0375787*(REAL(In[-6])+REAL(In[6]))
				              +   0.0482029*(REAL(In[-5])+REAL(In[5]))
				              +   0.0590941*(REAL(In[-4])+REAL(In[4]))
				              +   0.0692397*(REAL(In[-3])+REAL(In[3]))
				              +   0.0775364*(REAL(In[-2])+REAL(In[2]))
				              +   0.0829843*(REAL(In[-1])+REAL(In[1]))
				              +   0.0848841*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num51(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-19),-19,19,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num52 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num52(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.41176e-05*(REAL(In[-20])+REAL(In[20]))
				              +   3.29099e-05*(REAL(In[-19])+REAL(In[19]))
				              +   7.34588e-05*(REAL(In[-18])+REAL(In[18]))
				              +   0.000157004*(REAL(In[-17])+REAL(In[17]))
				              +   0.000321314*(REAL(In[-16])+REAL(In[16]))
				              +   0.000629649*(REAL(In[-15])+REAL(In[15]))
				              +   0.00118146*(REAL(In[-14])+REAL(In[14]))
				              +   0.00212269*(REAL(In[-13])+REAL(In[13]))
				              +   0.0036518*(REAL(In[-12])+REAL(In[12]))
				              +   0.00601557*(REAL(In[-11])+REAL(In[11]))
				              +   0.0094885*(REAL(In[-10])+REAL(In[10]))
				              +   0.0143307*(REAL(In[-9])+REAL(In[9]))
				              +   0.0207248*(REAL(In[-8])+REAL(In[8]))
				              +   0.0286987*(REAL(In[-7])+REAL(In[7]))
				              +   0.0380526*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483122*(REAL(In[-5])+REAL(In[5]))
				              +   0.0587327*(REAL(In[-4])+REAL(In[4]))
				              +   0.0683682*(REAL(In[-3])+REAL(In[3]))
				              +   0.076204*(REAL(In[-2])+REAL(In[2]))
				              +   0.0813303*(REAL(In[-1])+REAL(In[1]))
				              +   0.0831145*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num52(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-20),-20,20,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num53 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num53(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.96387e-05*(REAL(In[-20])+REAL(In[20]))
				              +   4.42415e-05*(REAL(In[-19])+REAL(In[19]))
				              +   9.56003e-05*(REAL(In[-18])+REAL(In[18]))
				              +   0.000198153*(REAL(In[-17])+REAL(In[17]))
				              +   0.000393961*(REAL(In[-16])+REAL(In[16]))
				              +   0.000751309*(REAL(In[-15])+REAL(In[15]))
				              +   0.00137434*(REAL(In[-14])+REAL(In[14]))
				              +   0.00241148*(REAL(In[-13])+REAL(In[13]))
				              +   0.00405868*(REAL(In[-12])+REAL(In[12]))
				              +   0.00655236*(REAL(In[-11])+REAL(In[11]))
				              +   0.0101467*(REAL(In[-10])+REAL(In[10]))
				              +   0.0150716*(REAL(In[-9])+REAL(In[9]))
				              +   0.0214738*(REAL(In[-8])+REAL(In[8]))
				              +   0.0293473*(REAL(In[-7])+REAL(In[7]))
				              +   0.0384717*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483755*(REAL(In[-5])+REAL(In[5]))
				              +   0.0583474*(REAL(In[-4])+REAL(In[4]))
				              +   0.067504*(REAL(In[-3])+REAL(In[3]))
				              +   0.0749117*(REAL(In[-2])+REAL(In[2]))
				              +   0.0797411*(REAL(In[-1])+REAL(In[1]))
				              +   0.0814191*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num53(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-20),-20,20,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num54 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num54(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.67671e-05*(REAL(In[-20])+REAL(In[20]))
				              +   5.83917e-05*(REAL(In[-19])+REAL(In[19]))
				              +   0.000122385*(REAL(In[-18])+REAL(In[18]))
				              +   0.000246454*(REAL(In[-17])+REAL(In[17]))
				              +   0.000476837*(REAL(In[-16])+REAL(In[16]))
				              +   0.000886405*(REAL(In[-15])+REAL(In[15]))
				              +   0.00158315*(REAL(In[-14])+REAL(In[14]))
				              +   0.0027167*(REAL(In[-13])+REAL(In[13]))
				              +   0.00447909*(REAL(In[-12])+REAL(In[12]))
				              +   0.0070952*(REAL(In[-11])+REAL(In[11]))
				              +   0.0107986*(REAL(In[-10])+REAL(In[10]))
				              +   0.0157907*(REAL(In[-9])+REAL(In[9]))
				              +   0.0221851*(REAL(In[-8])+REAL(In[8]))
				              +   0.0299467*(REAL(In[-7])+REAL(In[7]))
				              +   0.0388388*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483961*(REAL(In[-5])+REAL(In[5]))
				              +   0.0579406*(REAL(In[-4])+REAL(In[4]))
				              +   0.0666476*(REAL(In[-3])+REAL(In[3]))
				              +   0.073657*(REAL(In[-2])+REAL(In[2]))
				              +   0.0782117*(REAL(In[-1])+REAL(In[1]))
				              +   0.0797917*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num54(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-20),-20,20,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num55 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num55(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.62785e-05*(REAL(In[-21])+REAL(In[21]))
				              +   3.58021e-05*(REAL(In[-20])+REAL(In[20]))
				              +   7.57711e-05*(REAL(In[-19])+REAL(In[19]))
				              +   0.000154313*(REAL(In[-18])+REAL(In[18]))
				              +   0.000302415*(REAL(In[-17])+REAL(In[17]))
				              +   0.000570305*(REAL(In[-16])+REAL(In[16]))
				              +   0.00103494*(REAL(In[-15])+REAL(In[15]))
				              +   0.00180727*(REAL(In[-14])+REAL(In[14]))
				              +   0.00303694*(REAL(In[-13])+REAL(In[13]))
				              +   0.00491079*(REAL(In[-12])+REAL(In[12]))
				              +   0.00764133*(REAL(In[-11])+REAL(In[11]))
				              +   0.0114417*(REAL(In[-10])+REAL(In[10]))
				              +   0.0164859*(REAL(In[-9])+REAL(In[9]))
				              +   0.0228581*(REAL(In[-8])+REAL(In[8]))
				              +   0.0304978*(REAL(In[-7])+REAL(In[7]))
				              +   0.0391562*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483766*(REAL(In[-5])+REAL(In[5]))
				              +   0.0575139*(REAL(In[-4])+REAL(In[4]))
				              +   0.065798*(REAL(In[-3])+REAL(In[3]))
				              +   0.0724363*(REAL(In[-2])+REAL(In[2]))
				              +   0.0767365*(REAL(In[-1])+REAL(In[1]))
				              +   0.0782259*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num55(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-21),-21,21,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num56 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num56(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.20511e-05*(REAL(In[-21])+REAL(In[21]))
				              +   4.70635e-05*(REAL(In[-20])+REAL(In[20]))
				              +   9.68005e-05*(REAL(In[-19])+REAL(In[19]))
				              +   0.000191871*(REAL(In[-18])+REAL(In[18]))
				              +   0.000366505*(REAL(In[-17])+REAL(In[17]))
				              +   0.000674668*(REAL(In[-16])+REAL(In[16]))
				              +   0.00119685*(REAL(In[-15])+REAL(In[15]))
				              +   0.00204609*(REAL(In[-14])+REAL(In[14]))
				              +   0.00337094*(REAL(In[-13])+REAL(In[13]))
				              +   0.005352*(REAL(In[-12])+REAL(In[12]))
				              +   0.00818879*(REAL(In[-11])+REAL(In[11]))
				              +   0.0120743*(REAL(In[-10])+REAL(In[10]))
				              +   0.0171571*(REAL(In[-9])+REAL(In[9]))
				              +   0.0234944*(REAL(In[-8])+REAL(In[8]))
				              +   0.0310044*(REAL(In[-7])+REAL(In[7]))
				              +   0.0394294*(REAL(In[-6])+REAL(In[6]))
				              +   0.0483233*(REAL(In[-5])+REAL(In[5]))
				              +   0.0570732*(REAL(In[-4])+REAL(In[4]))
				              +   0.0649601*(REAL(In[-3])+REAL(In[3]))
				              +   0.0712524*(REAL(In[-2])+REAL(In[2]))
				              +   0.0753167*(REAL(In[-1])+REAL(In[1]))
				              +   0.0767223*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num56(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-21),-21,21,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num57 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num57(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.36497e-05*(REAL(In[-22])+REAL(In[22]))
				              +   2.93449e-05*(REAL(In[-21])+REAL(In[21]))
				              +   6.08806e-05*(REAL(In[-20])+REAL(In[20]))
				              +   0.000121889*(REAL(In[-19])+REAL(In[19]))
				              +   0.0002355*(REAL(In[-18])+REAL(In[18]))
				              +   0.000439091*(REAL(In[-17])+REAL(In[17]))
				              +   0.000790057*(REAL(In[-16])+REAL(In[16]))
				              +   0.00137183*(REAL(In[-15])+REAL(In[15]))
				              +   0.0022987*(REAL(In[-14])+REAL(In[14]))
				              +   0.0037171*(REAL(In[-13])+REAL(In[13]))
				              +   0.00580048*(REAL(In[-12])+REAL(In[12]))
				              +   0.00873502*(REAL(In[-11])+REAL(In[11]))
				              +   0.0126941*(REAL(In[-10])+REAL(In[10]))
				              +   0.0178025*(REAL(In[-9])+REAL(In[9]))
				              +   0.0240933*(REAL(In[-8])+REAL(In[8]))
				              +   0.0314669*(REAL(In[-7])+REAL(In[7]))
				              +   0.0396596*(REAL(In[-6])+REAL(In[6]))
				              +   0.0482373*(REAL(In[-5])+REAL(In[5]))
				              +   0.0566183*(REAL(In[-4])+REAL(In[4]))
				              +   0.0641313*(REAL(In[-3])+REAL(In[3]))
				              +   0.0701006*(REAL(In[-2])+REAL(In[2]))
				              +   0.0739457*(REAL(In[-1])+REAL(In[1]))
				              +   0.0752737*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num57(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-22),-22,22,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num58 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num58(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.83782e-05*(REAL(In[-22])+REAL(In[22]))
				              +   3.8416e-05*(REAL(In[-21])+REAL(In[21]))
				              +   7.75942e-05*(REAL(In[-20])+REAL(In[20]))
				              +   0.000151444*(REAL(In[-19])+REAL(In[19]))
				              +   0.000285616*(REAL(In[-18])+REAL(In[18]))
				              +   0.000520497*(REAL(In[-17])+REAL(In[17]))
				              +   0.000916561*(REAL(In[-16])+REAL(In[16]))
				              +   0.00155959*(REAL(In[-15])+REAL(In[15]))
				              +   0.00256429*(REAL(In[-14])+REAL(In[14]))
				              +   0.00407407*(REAL(In[-13])+REAL(In[13]))
				              +   0.00625457*(REAL(In[-12])+REAL(In[12]))
				              +   0.00927839*(REAL(In[-11])+REAL(In[11]))
				              +   0.0133001*(REAL(In[-10])+REAL(In[10]))
				              +   0.0184222*(REAL(In[-9])+REAL(In[9]))
				              +   0.0246568*(REAL(In[-8])+REAL(In[8]))
				              +   0.0318888*(REAL(In[-7])+REAL(In[7]))
				              +   0.0398517*(REAL(In[-6])+REAL(In[6]))
				              +   0.0481239*(REAL(In[-5])+REAL(In[5]))
				              +   0.0561541*(REAL(In[-4])+REAL(In[4]))
				              +   0.0633153*(REAL(In[-3])+REAL(In[3]))
				              +   0.068983*(REAL(In[-2])+REAL(In[2]))
				              +   0.0726244*(REAL(In[-1])+REAL(In[1]))
				              +   0.0738804*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num58(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-22),-22,22,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num59 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num59(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.43338e-05*(REAL(In[-22])+REAL(In[22]))
				              +   4.95315e-05*(REAL(In[-21])+REAL(In[21]))
				              +   9.75432e-05*(REAL(In[-20])+REAL(In[20]))
				              +   0.000185847*(REAL(In[-19])+REAL(In[19]))
				              +   0.000342576*(REAL(In[-18])+REAL(In[18]))
				              +   0.000610945*(REAL(In[-17])+REAL(In[17]))
				              +   0.00105412*(REAL(In[-16])+REAL(In[16]))
				              +   0.00175963*(REAL(In[-15])+REAL(In[15]))
				              +   0.00284181*(REAL(In[-14])+REAL(In[14]))
				              +   0.0044403*(REAL(In[-13])+REAL(In[13]))
				              +   0.00671232*(REAL(In[-12])+REAL(In[12]))
				              +   0.00981695*(REAL(In[-11])+REAL(In[11]))
				              +   0.0138907*(REAL(In[-10])+REAL(In[10]))
				              +   0.0190158*(REAL(In[-9])+REAL(In[9]))
				              +   0.0251853*(REAL(In[-8])+REAL(In[8]))
				              +   0.0322718*(REAL(In[-7])+REAL(In[7]))
				              +   0.0400076*(REAL(In[-6])+REAL(In[6]))
				              +   0.047985*(REAL(In[-5])+REAL(In[5]))
				              +   0.0556815*(REAL(In[-4])+REAL(In[4]))
				              +   0.0625115*(REAL(In[-3])+REAL(In[3]))
				              +   0.0678972*(REAL(In[-2])+REAL(In[2]))
				              +   0.0713489*(REAL(In[-1])+REAL(In[1]))
				              +   0.072538*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num59(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-22),-22,22,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num60 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num60(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.54796e-05*(REAL(In[-23])+REAL(In[23]))
				              +   3.17215e-05*(REAL(In[-22])+REAL(In[22]))
				              +   6.29652e-05*(REAL(In[-21])+REAL(In[21]))
				              +   0.000121059*(REAL(In[-20])+REAL(In[20]))
				              +   0.000225449*(REAL(In[-19])+REAL(In[19]))
				              +   0.000406675*(REAL(In[-18])+REAL(In[18]))
				              +   0.000710558*(REAL(In[-17])+REAL(In[17]))
				              +   0.00120255*(REAL(In[-16])+REAL(In[16]))
				              +   0.00197132*(REAL(In[-15])+REAL(In[15]))
				              +   0.00313014*(REAL(In[-14])+REAL(In[14]))
				              +   0.00481416*(REAL(In[-13])+REAL(In[13]))
				              +   0.00717181*(REAL(In[-12])+REAL(In[12]))
				              +   0.0103488*(REAL(In[-11])+REAL(In[11]))
				              +   0.0144644*(REAL(In[-10])+REAL(In[10]))
				              +   0.0195822*(REAL(In[-9])+REAL(In[9]))
				              +   0.0256789*(REAL(In[-8])+REAL(In[8]))
				              +   0.0326168*(REAL(In[-7])+REAL(In[7]))
				              +   0.0401289*(REAL(In[-6])+REAL(In[6]))
				              +   0.0478217*(REAL(In[-5])+REAL(In[5]))
				              +   0.0552006*(REAL(In[-4])+REAL(In[4]))
				              +   0.0617184*(REAL(In[-3])+REAL(In[3]))
				              +   0.06684*(REAL(In[-2])+REAL(In[2]))
				              +   0.0701147*(REAL(In[-1])+REAL(In[1]))
				              +   0.0712416*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num60(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-23),-23,23,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num61 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num61(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.03927e-05*(REAL(In[-23])+REAL(In[23]))
				              +   4.07598e-05*(REAL(In[-22])+REAL(In[22]))
				              +   7.89993e-05*(REAL(In[-21])+REAL(In[21]))
				              +   0.000148473*(REAL(In[-20])+REAL(In[20]))
				              +   0.000270585*(REAL(In[-19])+REAL(In[19]))
				              +   0.000478183*(REAL(In[-18])+REAL(In[18]))
				              +   0.000819439*(REAL(In[-17])+REAL(In[17]))
				              +   0.00136167*(REAL(In[-16])+REAL(In[16]))
				              +   0.00219412*(REAL(In[-15])+REAL(In[15]))
				              +   0.00342833*(REAL(In[-14])+REAL(In[14]))
				              +   0.00519443*(REAL(In[-13])+REAL(In[13]))
				              +   0.00763178*(REAL(In[-12])+REAL(In[12]))
				              +   0.0108729*(REAL(In[-11])+REAL(In[11]))
				              +   0.015021*(REAL(In[-10])+REAL(In[10]))
				              +   0.0201227*(REAL(In[-9])+REAL(In[9]))
				              +   0.0261401*(REAL(In[-8])+REAL(In[8]))
				              +   0.0329275*(REAL(In[-7])+REAL(In[7]))
				              +   0.0402202*(REAL(In[-6])+REAL(In[6]))
				              +   0.0476391*(REAL(In[-5])+REAL(In[5]))
				              +   0.0547161*(REAL(In[-4])+REAL(In[4]))
				              +   0.0609396*(REAL(In[-3])+REAL(In[3]))
				              +   0.0658138*(REAL(In[-2])+REAL(In[2]))
				              +   0.0689235*(REAL(In[-1])+REAL(In[1]))
				              +   0.0699924*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num61(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-23),-23,23,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num62 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num62(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.31642e-05*(REAL(In[-24])+REAL(In[24]))
				              +   2.64716e-05*(REAL(In[-23])+REAL(In[23]))
				              +   5.16722e-05*(REAL(In[-22])+REAL(In[22]))
				              +   9.79092e-05*(REAL(In[-21])+REAL(In[21]))
				              +   0.000180086*(REAL(In[-20])+REAL(In[20]))
				              +   0.000321533*(REAL(In[-19])+REAL(In[19]))
				              +   0.000557266*(REAL(In[-18])+REAL(In[18]))
				              +   0.000937537*(REAL(In[-17])+REAL(In[17]))
				              +   0.0015311*(REAL(In[-16])+REAL(In[16]))
				              +   0.00242723*(REAL(In[-15])+REAL(In[15]))
				              +   0.00373513*(REAL(In[-14])+REAL(In[14]))
				              +   0.00557946*(REAL(In[-13])+REAL(In[13]))
				              +   0.00809036*(REAL(In[-12])+REAL(In[12]))
				              +   0.0113876*(REAL(In[-11])+REAL(In[11]))
				              +   0.0155593*(REAL(In[-10])+REAL(In[10]))
				              +   0.0206364*(REAL(In[-9])+REAL(In[9]))
				              +   0.0265687*(REAL(In[-8])+REAL(In[8]))
				              +   0.0332044*(REAL(In[-7])+REAL(In[7]))
				              +   0.040282*(REAL(In[-6])+REAL(In[6]))
				              +   0.0474369*(REAL(In[-5])+REAL(In[5]))
				              +   0.0542265*(REAL(In[-4])+REAL(In[4]))
				              +   0.0601723*(REAL(In[-3])+REAL(In[3]))
				              +   0.0648145*(REAL(In[-2])+REAL(In[2]))
				              +   0.06777*(REAL(In[-1])+REAL(In[1]))
				              +   0.0687848*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num62(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-24),-24,24,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num63 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num63(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.72561e-05*(REAL(In[-24])+REAL(In[24]))
				              +   3.38948e-05*(REAL(In[-23])+REAL(In[23]))
				              +   6.46913e-05*(REAL(In[-22])+REAL(In[22]))
				              +   0.000119973*(REAL(In[-21])+REAL(In[21]))
				              +   0.000216194*(REAL(In[-20])+REAL(In[20]))
				              +   0.000378555*(REAL(In[-19])+REAL(In[19]))
				              +   0.000644076*(REAL(In[-18])+REAL(In[18]))
				              +   0.0010648*(REAL(In[-17])+REAL(In[17]))
				              +   0.00171051*(REAL(In[-16])+REAL(In[16]))
				              +   0.00266997*(REAL(In[-15])+REAL(In[15]))
				              +   0.00404958*(REAL(In[-14])+REAL(In[14]))
				              +   0.00596813*(REAL(In[-13])+REAL(In[13]))
				              +   0.00854653*(REAL(In[-12])+REAL(In[12]))
				              +   0.0118923*(REAL(In[-11])+REAL(In[11]))
				              +   0.0160792*(REAL(In[-10])+REAL(In[10]))
				              +   0.0211246*(REAL(In[-9])+REAL(In[9]))
				              +   0.0269672*(REAL(In[-8])+REAL(In[8]))
				              +   0.0334508*(REAL(In[-7])+REAL(In[7]))
				              +   0.0403182*(REAL(In[-6])+REAL(In[6]))
				              +   0.0472194*(REAL(In[-5])+REAL(In[5]))
				              +   0.0537357*(REAL(In[-4])+REAL(In[4]))
				              +   0.0594196*(REAL(In[-3])+REAL(In[3]))
				              +   0.0638439*(REAL(In[-2])+REAL(In[2]))
				              +   0.0666552*(REAL(In[-1])+REAL(In[1]))
				              +   0.0676195*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num63(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-24),-24,24,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num64 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num64(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.2306e-05*(REAL(In[-24])+REAL(In[24]))
				              +   4.28469e-05*(REAL(In[-23])+REAL(In[23]))
				              +   8.00486e-05*(REAL(In[-22])+REAL(In[22]))
				              +   0.000145453*(REAL(In[-21])+REAL(In[21]))
				              +   0.000257058*(REAL(In[-20])+REAL(In[20]))
				              +   0.000441849*(REAL(In[-19])+REAL(In[19]))
				              +   0.000738674*(REAL(In[-18])+REAL(In[18]))
				              +   0.00120107*(REAL(In[-17])+REAL(In[17]))
				              +   0.00189941*(REAL(In[-16])+REAL(In[16]))
				              +   0.00292151*(REAL(In[-15])+REAL(In[15]))
				              +   0.0043705*(REAL(In[-14])+REAL(In[14]))
				              +   0.00635905*(REAL(In[-13])+REAL(In[13]))
				              +   0.00899889*(REAL(In[-12])+REAL(In[12]))
				              +   0.0123857*(REAL(In[-11])+REAL(In[11]))
				              +   0.0165802*(REAL(In[-10])+REAL(In[10]))
				              +   0.0215872*(REAL(In[-9])+REAL(In[9]))
				              +   0.0273362*(REAL(In[-8])+REAL(In[8]))
				              +   0.0336679*(REAL(In[-7])+REAL(In[7]))
				              +   0.0403302*(REAL(In[-6])+REAL(In[6]))
				              +   0.0469874*(REAL(In[-5])+REAL(In[5]))
				              +   0.0532437*(REAL(In[-4])+REAL(In[4]))
				              +   0.0586801*(REAL(In[-3])+REAL(In[3]))
				              +   0.0628999*(REAL(In[-2])+REAL(In[2]))
				              +   0.0655761*(REAL(In[-1])+REAL(In[1]))
				              +   0.0664933*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num64(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-24),-24,24,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num65 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num65(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.47321e-05*(REAL(In[-25])+REAL(In[25]))
				              +   2.84588e-05*(REAL(In[-24])+REAL(In[24]))
				              +   5.35174e-05*(REAL(In[-23])+REAL(In[23]))
				              +   9.79722e-05*(REAL(In[-22])+REAL(In[22]))
				              +   0.000174598*(REAL(In[-21])+REAL(In[21]))
				              +   0.000302903*(REAL(In[-20])+REAL(In[20]))
				              +   0.00051156*(REAL(In[-19])+REAL(In[19]))
				              +   0.000841044*(REAL(In[-18])+REAL(In[18]))
				              +   0.00134607*(REAL(In[-17])+REAL(In[17]))
				              +   0.00209724*(REAL(In[-16])+REAL(In[16]))
				              +   0.00318094*(REAL(In[-15])+REAL(In[15]))
				              +   0.00469668*(REAL(In[-14])+REAL(In[14]))
				              +   0.0067508*(REAL(In[-13])+REAL(In[13]))
				              +   0.00944601*(REAL(In[-12])+REAL(In[12]))
				              +   0.0128668*(REAL(In[-11])+REAL(In[11]))
				              +   0.0170616*(REAL(In[-10])+REAL(In[10]))
				              +   0.0220241*(REAL(In[-9])+REAL(In[9]))
				              +   0.0276762*(REAL(In[-8])+REAL(In[8]))
				              +   0.0338565*(REAL(In[-7])+REAL(In[7]))
				              +   0.0403187*(REAL(In[-6])+REAL(In[6]))
				              +   0.0467412*(REAL(In[-5])+REAL(In[5]))
				              +   0.0527498*(REAL(In[-4])+REAL(In[4]))
				              +   0.0579524*(REAL(In[-3])+REAL(In[3]))
				              +   0.0619797*(REAL(In[-2])+REAL(In[2]))
				              +   0.0645293*(REAL(In[-1])+REAL(In[1]))
				              +   0.0654022*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num65(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-25),-25,25,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num66 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num66(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.89634e-05*(REAL(In[-25])+REAL(In[25]))
				              +   3.58687e-05*(REAL(In[-24])+REAL(In[24]))
				              +   6.61026e-05*(REAL(In[-23])+REAL(In[23]))
				              +   0.000118692*(REAL(In[-22])+REAL(In[22]))
				              +   0.000207649*(REAL(In[-21])+REAL(In[21]))
				              +   0.000353947*(REAL(In[-20])+REAL(In[20]))
				              +   0.000587826*(REAL(In[-19])+REAL(In[19]))
				              +   0.000951178*(REAL(In[-18])+REAL(In[18]))
				              +   0.0014996*(REAL(In[-17])+REAL(In[17]))
				              +   0.00230353*(REAL(In[-16])+REAL(In[16]))
				              +   0.00344756*(REAL(In[-15])+REAL(In[15]))
				              +   0.00502728*(REAL(In[-14])+REAL(In[14]))
				              +   0.00714259*(REAL(In[-13])+REAL(In[13]))
				              +   0.00988736*(REAL(In[-12])+REAL(In[12]))
				              +   0.0133354*(REAL(In[-11])+REAL(In[11]))
				              +   0.0175241*(REAL(In[-10])+REAL(In[10]))
				              +   0.0224371*(REAL(In[-9])+REAL(In[9]))
				              +   0.0279898*(REAL(In[-8])+REAL(In[8]))
				              +   0.03402*(REAL(In[-7])+REAL(In[7]))
				              +   0.0402876*(REAL(In[-6])+REAL(In[6]))
				              +   0.0464847*(REAL(In[-5])+REAL(In[5]))
				              +   0.0522578*(REAL(In[-4])+REAL(In[4]))
				              +   0.0572393*(REAL(In[-3])+REAL(In[3]))
				              +   0.0610856*(REAL(In[-2])+REAL(In[2]))
				              +   0.0635164*(REAL(In[-1])+REAL(In[1]))
				              +   0.064348*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num66(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-25),-25,25,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num67 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num67(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.26803e-05*(REAL(In[-26])+REAL(In[26]))
				              +   2.41076e-05*(REAL(In[-25])+REAL(In[25]))
				              +   4.46928e-05*(REAL(In[-24])+REAL(In[24]))
				              +   8.07938e-05*(REAL(In[-23])+REAL(In[23]))
				              +   0.000142422*(REAL(In[-22])+REAL(In[22]))
				              +   0.000244812*(REAL(In[-21])+REAL(In[21]))
				              +   0.000410344*(REAL(In[-20])+REAL(In[20]))
				              +   0.000670687*(REAL(In[-19])+REAL(In[19]))
				              +   0.00106893*(REAL(In[-18])+REAL(In[18]))
				              +   0.00166126*(REAL(In[-17])+REAL(In[17]))
				              +   0.00251758*(REAL(In[-16])+REAL(In[16]))
				              +   0.00372038*(REAL(In[-15])+REAL(In[15]))
				              +   0.00536103*(REAL(In[-14])+REAL(In[14]))
				              +   0.007533*(REAL(In[-13])+REAL(In[13]))
				              +   0.0103215*(REAL(In[-12])+REAL(In[12]))
				              +   0.0137905*(REAL(In[-11])+REAL(In[11]))
				              +   0.0179669*(REAL(In[-10])+REAL(In[10]))
				              +   0.0228256*(REAL(In[-9])+REAL(In[9]))
				              +   0.0282769*(REAL(In[-8])+REAL(In[8]))
				              +   0.0341584*(REAL(In[-7])+REAL(In[7]))
				              +   0.0402366*(REAL(In[-6])+REAL(In[6]))
				              +   0.0462172*(REAL(In[-5])+REAL(In[5]))
				              +   0.0517658*(REAL(In[-4])+REAL(In[4]))
				              +   0.0565381*(REAL(In[-3])+REAL(In[3]))
				              +   0.0602138*(REAL(In[-2])+REAL(In[2]))
				              +   0.062533*(REAL(In[-1])+REAL(In[1]))
				              +   0.0633258*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num67(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-26),-26,26,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num68 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num68(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.62543e-05*(REAL(In[-26])+REAL(In[26]))
				              +   3.02929e-05*(REAL(In[-25])+REAL(In[25]))
				              +   5.50949e-05*(REAL(In[-24])+REAL(In[24]))
				              +   9.77867e-05*(REAL(In[-23])+REAL(In[23]))
				              +   0.000169373*(REAL(In[-22])+REAL(In[22]))
				              +   0.000286291*(REAL(In[-21])+REAL(In[21]))
				              +   0.000472244*(REAL(In[-20])+REAL(In[20]))
				              +   0.000760191*(REAL(In[-19])+REAL(In[19]))
				              +   0.0011942*(REAL(In[-18])+REAL(In[18]))
				              +   0.00183074*(REAL(In[-17])+REAL(In[17]))
				              +   0.00273889*(REAL(In[-16])+REAL(In[16]))
				              +   0.00399871*(REAL(In[-15])+REAL(In[15]))
				              +   0.0056972*(REAL(In[-14])+REAL(In[14]))
				              +   0.00792137*(REAL(In[-13])+REAL(In[13]))
				              +   0.0107482*(REAL(In[-12])+REAL(In[12]))
				              +   0.0142321*(REAL(In[-11])+REAL(In[11]))
				              +   0.0183908*(REAL(In[-10])+REAL(In[10]))
				              +   0.0231914*(REAL(In[-9])+REAL(In[9]))
				              +   0.0285399*(REAL(In[-8])+REAL(In[8]))
				              +   0.0342747*(REAL(In[-7])+REAL(In[7]))
				              +   0.0401692*(REAL(In[-6])+REAL(In[6]))
				              +   0.0459419*(REAL(In[-5])+REAL(In[5]))
				              +   0.0512769*(REAL(In[-4])+REAL(In[4]))
				              +   0.0558511*(REAL(In[-3])+REAL(In[3]))
				              +   0.0593662*(REAL(In[-2])+REAL(In[2]))
				              +   0.0615805*(REAL(In[-1])+REAL(In[1]))
				              +   0.0623369*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num68(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-26),-26,26,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num69 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num69(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2.05902e-05*(REAL(In[-26])+REAL(In[26]))
				              +   3.76512e-05*(REAL(In[-25])+REAL(In[25]))
				              +   6.72385e-05*(REAL(In[-24])+REAL(In[24]))
				              +   0.000117268*(REAL(In[-23])+REAL(In[23]))
				              +   0.000199738*(REAL(In[-22])+REAL(In[22]))
				              +   0.000332247*(REAL(In[-21])+REAL(In[21]))
				              +   0.00053974*(REAL(In[-20])+REAL(In[20]))
				              +   0.000856304*(REAL(In[-19])+REAL(In[19]))
				              +   0.00132676*(REAL(In[-18])+REAL(In[18]))
				              +   0.0020076*(REAL(In[-17])+REAL(In[17]))
				              +   0.00296677*(REAL(In[-16])+REAL(In[16]))
				              +   0.00428165*(REAL(In[-15])+REAL(In[15]))
				              +   0.00603474*(REAL(In[-14])+REAL(In[14]))
				              +   0.00830668*(REAL(In[-13])+REAL(In[13]))
				              +   0.0111665*(REAL(In[-12])+REAL(In[12]))
				              +   0.0146598*(REAL(In[-11])+REAL(In[11]))
				              +   0.0187957*(REAL(In[-10])+REAL(In[10]))
				              +   0.0235348*(REAL(In[-9])+REAL(In[9]))
				              +   0.0287795*(REAL(In[-8])+REAL(In[8]))
				              +   0.0343698*(REAL(In[-7])+REAL(In[7]))
				              +   0.0400859*(REAL(In[-6])+REAL(In[6]))
				              +   0.045659*(REAL(In[-5])+REAL(In[5]))
				              +   0.0507905*(REAL(In[-4])+REAL(In[4]))
				              +   0.0551772*(REAL(In[-3])+REAL(In[3]))
				              +   0.0585407*(REAL(In[-2])+REAL(In[2]))
				              +   0.0606564*(REAL(In[-1])+REAL(In[1]))
				              +   0.0613785*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num69(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-26),-26,26,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num70 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num70(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.40379e-05*(REAL(In[-27])+REAL(In[27]))
				              +   2.57934e-05*(REAL(In[-26])+REAL(In[26]))
				              +   4.63175e-05*(REAL(In[-25])+REAL(In[25]))
				              +   8.12851e-05*(REAL(In[-24])+REAL(In[24]))
				              +   0.000139414*(REAL(In[-23])+REAL(In[23]))
				              +   0.000233686*(REAL(In[-22])+REAL(In[22]))
				              +   0.000382814*(REAL(In[-21])+REAL(In[21]))
				              +   0.000612877*(REAL(In[-20])+REAL(In[20]))
				              +   0.000958935*(REAL(In[-19])+REAL(In[19]))
				              +   0.00146634*(REAL(In[-18])+REAL(In[18]))
				              +   0.00219134*(REAL(In[-17])+REAL(In[17]))
				              +   0.00320048*(REAL(In[-16])+REAL(In[16]))
				              +   0.00456826*(REAL(In[-15])+REAL(In[15]))
				              +   0.0063726*(REAL(In[-14])+REAL(In[14]))
				              +   0.00868784*(REAL(In[-13])+REAL(In[13]))
				              +   0.0115754*(REAL(In[-12])+REAL(In[12]))
				              +   0.0150728*(REAL(In[-11])+REAL(In[11]))
				              +   0.0191813*(REAL(In[-10])+REAL(In[10]))
				              +   0.0238558*(REAL(In[-9])+REAL(In[9]))
				              +   0.028996*(REAL(In[-8])+REAL(In[8]))
				              +   0.034444*(REAL(In[-7])+REAL(In[7]))
				              +   0.039987*(REAL(In[-6])+REAL(In[6]))
				              +   0.0453685*(REAL(In[-5])+REAL(In[5]))
				              +   0.0503059*(REAL(In[-4])+REAL(In[4]))
				              +   0.0545148*(REAL(In[-3])+REAL(In[3]))
				              +   0.057735*(REAL(In[-2])+REAL(In[2]))
				              +   0.0597578*(REAL(In[-1])+REAL(In[1]))
				              +   0.0604476*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num70(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-27),-27,27,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num71 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num71(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.77193e-05*(REAL(In[-27])+REAL(In[27]))
				              +   3.19761e-05*(REAL(In[-26])+REAL(In[26]))
				              +   5.64324e-05*(REAL(In[-25])+REAL(In[25]))
				              +   9.73995e-05*(REAL(In[-24])+REAL(In[24]))
				              +   0.000164403*(REAL(In[-23])+REAL(In[23]))
				              +   0.000271388*(REAL(In[-22])+REAL(In[22]))
				              +   0.000438122*(REAL(In[-21])+REAL(In[21]))
				              +   0.000691712*(REAL(In[-20])+REAL(In[20]))
				              +   0.00106802*(REAL(In[-19])+REAL(In[19]))
				              +   0.00161273*(REAL(In[-18])+REAL(In[18]))
				              +   0.00238159*(REAL(In[-17])+REAL(In[17]))
				              +   0.00343953*(REAL(In[-16])+REAL(In[16]))
				              +   0.00485798*(REAL(In[-15])+REAL(In[15]))
				              +   0.00671024*(REAL(In[-14])+REAL(In[14]))
				              +   0.00906454*(REAL(In[-13])+REAL(In[13]))
				              +   0.0119751*(REAL(In[-12])+REAL(In[12]))
				              +   0.0154717*(REAL(In[-11])+REAL(In[11]))
				              +   0.0195488*(REAL(In[-10])+REAL(In[10]))
				              +   0.0241563*(REAL(In[-9])+REAL(In[9]))
				              +   0.029192*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345004*(REAL(In[-7])+REAL(In[7]))
				              +   0.0398758*(REAL(In[-6])+REAL(In[6]))
				              +   0.0450733*(REAL(In[-5])+REAL(In[5]))
				              +   0.0498259*(REAL(In[-4])+REAL(In[4]))
				              +   0.0538662*(REAL(In[-3])+REAL(In[3]))
				              +   0.0569513*(REAL(In[-2])+REAL(In[2]))
				              +   0.0588864*(REAL(In[-1])+REAL(In[1]))
				              +   0.059546*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num71(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-27),-27,27,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num72 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num72(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.22089e-05*(REAL(In[-28])+REAL(In[28]))
				              +   2.21291e-05*(REAL(In[-27])+REAL(In[27]))
				              +   3.92514e-05*(REAL(In[-26])+REAL(In[26]))
				              +   6.81328e-05*(REAL(In[-25])+REAL(In[25]))
				              +   0.000115735*(REAL(In[-24])+REAL(In[24]))
				              +   0.000192389*(REAL(In[-23])+REAL(In[23]))
				              +   0.000312972*(REAL(In[-22])+REAL(In[22]))
				              +   0.000498238*(REAL(In[-21])+REAL(In[21]))
				              +   0.000776205*(REAL(In[-20])+REAL(In[20]))
				              +   0.00118338*(REAL(In[-19])+REAL(In[19]))
				              +   0.00176555*(REAL(In[-18])+REAL(In[18]))
				              +   0.00257776*(REAL(In[-17])+REAL(In[17]))
				              +   0.0036831*(REAL(In[-16])+REAL(In[16]))
				              +   0.00514983*(REAL(In[-15])+REAL(In[15]))
				              +   0.00704659*(REAL(In[-14])+REAL(In[14]))
				              +   0.00943569*(REAL(In[-13])+REAL(In[13]))
				              +   0.0123645*(REAL(In[-12])+REAL(In[12]))
				              +   0.0158557*(REAL(In[-11])+REAL(In[11]))
				              +   0.0198977*(REAL(In[-10])+REAL(In[10]))
				              +   0.024436*(REAL(In[-9])+REAL(In[9]))
				              +   0.0293672*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345386*(REAL(In[-7])+REAL(In[7]))
				              +   0.0397515*(REAL(In[-6])+REAL(In[6]))
				              +   0.0447724*(REAL(In[-5])+REAL(In[5]))
				              +   0.0493487*(REAL(In[-4])+REAL(In[4]))
				              +   0.053229*(REAL(In[-3])+REAL(In[3]))
				              +   0.0561861*(REAL(In[-2])+REAL(In[2]))
				              +   0.0580386*(REAL(In[-1])+REAL(In[1]))
				              +   0.0586696*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num72(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-28),-28,28,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num73 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num73(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.53562e-05*(REAL(In[-28])+REAL(In[28]))
				              +   2.73614e-05*(REAL(In[-27])+REAL(In[27]))
				              +   4.77385e-05*(REAL(In[-26])+REAL(In[26]))
				              +   8.156e-05*(REAL(In[-25])+REAL(In[25]))
				              +   0.000136447*(REAL(In[-24])+REAL(In[24]))
				              +   0.000223527*(REAL(In[-23])+REAL(In[23]))
				              +   0.000358569*(REAL(In[-22])+REAL(In[22]))
				              +   0.00056324*(REAL(In[-21])+REAL(In[21]))
				              +   0.000866349*(REAL(In[-20])+REAL(In[20]))
				              +   0.00130488*(REAL(In[-19])+REAL(In[19]))
				              +   0.00192454*(REAL(In[-18])+REAL(In[18]))
				              +   0.00277945*(REAL(In[-17])+REAL(In[17]))
				              +   0.00393071*(REAL(In[-16])+REAL(In[16]))
				              +   0.00544328*(REAL(In[-15])+REAL(In[15]))
				              +   0.00738123*(REAL(In[-14])+REAL(In[14]))
				              +   0.00980109*(REAL(In[-13])+REAL(In[13]))
				              +   0.0127438*(REAL(In[-12])+REAL(In[12]))
				              +   0.0162256*(REAL(In[-11])+REAL(In[11]))
				              +   0.0202293*(REAL(In[-10])+REAL(In[10]))
				              +   0.0246967*(REAL(In[-9])+REAL(In[9]))
				              +   0.029524*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345613*(REAL(In[-7])+REAL(In[7]))
				              +   0.039617*(REAL(In[-6])+REAL(In[6]))
				              +   0.0444685*(REAL(In[-5])+REAL(In[5]))
				              +   0.0488766*(REAL(In[-4])+REAL(In[4]))
				              +   0.0526051*(REAL(In[-3])+REAL(In[3]))
				              +   0.0554412*(REAL(In[-2])+REAL(In[2]))
				              +   0.0572157*(REAL(In[-1])+REAL(In[1]))
				              +   0.0578198*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num73(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-28),-28,28,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num74 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num74(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.91195e-05*(REAL(In[-28])+REAL(In[28]))
				              +   3.3513e-05*(REAL(In[-27])+REAL(In[27]))
				              +   5.75554e-05*(REAL(In[-26])+REAL(In[26]))
				              +   9.68494e-05*(REAL(In[-25])+REAL(In[25]))
				              +   0.000159678*(REAL(In[-24])+REAL(In[24]))
				              +   0.000257946*(REAL(In[-23])+REAL(In[23]))
				              +   0.000408272*(REAL(In[-22])+REAL(In[22]))
				              +   0.00063315*(REAL(In[-21])+REAL(In[21]))
				              +   0.000962059*(REAL(In[-20])+REAL(In[20]))
				              +   0.0014323*(REAL(In[-19])+REAL(In[19]))
				              +   0.0020893*(REAL(In[-18])+REAL(In[18]))
				              +   0.00298611*(REAL(In[-17])+REAL(In[17]))
				              +   0.00418166*(REAL(In[-16])+REAL(In[16]))
				              +   0.00573756*(REAL(In[-15])+REAL(In[15]))
				              +   0.00771335*(REAL(In[-14])+REAL(In[14]))
				              +   0.01016*(REAL(In[-13])+REAL(In[13]))
				              +   0.0131125*(REAL(In[-12])+REAL(In[12]))
				              +   0.016581*(REAL(In[-11])+REAL(In[11]))
				              +   0.0205435*(REAL(In[-10])+REAL(In[10]))
				              +   0.0249387*(REAL(In[-9])+REAL(In[9]))
				              +   0.0296627*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345688*(REAL(In[-7])+REAL(In[7]))
				              +   0.0394726*(REAL(In[-6])+REAL(In[6]))
				              +   0.0441614*(REAL(In[-5])+REAL(In[5]))
				              +   0.0484091*(REAL(In[-4])+REAL(In[4]))
				              +   0.0519934*(REAL(In[-3])+REAL(In[3]))
				              +   0.0547149*(REAL(In[-2])+REAL(In[2]))
				              +   0.0564158*(REAL(In[-1])+REAL(In[1]))
				              +   0.0569944*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num74(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-28),-28,28,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num75 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num75(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.33954e-05*(REAL(In[-29])+REAL(In[29]))
				              +   2.35771e-05*(REAL(In[-28])+REAL(In[28]))
				              +   4.06826e-05*(REAL(In[-27])+REAL(In[27]))
				              +   6.88197e-05*(REAL(In[-26])+REAL(In[26]))
				              +   0.00011413*(REAL(In[-25])+REAL(In[25]))
				              +   0.000185556*(REAL(In[-24])+REAL(In[24]))
				              +   0.000295756*(REAL(In[-23])+REAL(In[23]))
				              +   0.000462143*(REAL(In[-22])+REAL(In[22]))
				              +   0.000707953*(REAL(In[-21])+REAL(In[21]))
				              +   0.0010632*(REAL(In[-20])+REAL(In[20]))
				              +   0.00156536*(REAL(In[-19])+REAL(In[19]))
				              +   0.00225942*(REAL(In[-18])+REAL(In[18]))
				              +   0.00319715*(REAL(In[-17])+REAL(In[17]))
				              +   0.00443521*(REAL(In[-16])+REAL(In[16]))
				              +   0.00603185*(REAL(In[-15])+REAL(In[15]))
				              +   0.00804214*(REAL(In[-14])+REAL(In[14]))
				              +   0.0105118*(REAL(In[-13])+REAL(In[13]))
				              +   0.01347*(REAL(In[-12])+REAL(In[12]))
				              +   0.0169217*(REAL(In[-11])+REAL(In[11]))
				              +   0.0208403*(REAL(In[-10])+REAL(In[10]))
				              +   0.0251622*(REAL(In[-9])+REAL(In[9]))
				              +   0.0297836*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345615*(REAL(In[-7])+REAL(In[7]))
				              +   0.039318*(REAL(In[-6])+REAL(In[6]))
				              +   0.0438506*(REAL(In[-5])+REAL(In[5]))
				              +   0.047945*(REAL(In[-4])+REAL(In[4]))
				              +   0.0513922*(REAL(In[-3])+REAL(In[3]))
				              +   0.0540051*(REAL(In[-2])+REAL(In[2]))
				              +   0.0556363*(REAL(In[-1])+REAL(In[1]))
				              +   0.0561909*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num75(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-29),-29,29,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num76 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num76(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.66271e-05*(REAL(In[-29])+REAL(In[29]))
				              +   2.88123e-05*(REAL(In[-28])+REAL(In[28]))
				              +   4.89736e-05*(REAL(In[-27])+REAL(In[27]))
				              +   8.16523e-05*(REAL(In[-26])+REAL(In[26]))
				              +   0.000133536*(REAL(In[-25])+REAL(In[25]))
				              +   0.000214214*(REAL(In[-24])+REAL(In[24]))
				              +   0.000337071*(REAL(In[-23])+REAL(In[23]))
				              +   0.000520257*(REAL(In[-22])+REAL(In[22]))
				              +   0.000787655*(REAL(In[-21])+REAL(In[21]))
				              +   0.00116971*(REAL(In[-20])+REAL(In[20]))
				              +   0.00170388*(REAL(In[-19])+REAL(In[19]))
				              +   0.00243459*(REAL(In[-18])+REAL(In[18]))
				              +   0.00341219*(REAL(In[-17])+REAL(In[17]))
				              +   0.00469098*(REAL(In[-16])+REAL(In[16]))
				              +   0.00632581*(REAL(In[-15])+REAL(In[15]))
				              +   0.00836741*(REAL(In[-14])+REAL(In[14]))
				              +   0.0108565*(REAL(In[-13])+REAL(In[13]))
				              +   0.0138168*(REAL(In[-12])+REAL(In[12]))
				              +   0.0172485*(REAL(In[-11])+REAL(In[11]))
				              +   0.0211211*(REAL(In[-10])+REAL(In[10]))
				              +   0.025369*(REAL(In[-9])+REAL(In[9]))
				              +   0.0298891*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345418*(REAL(In[-7])+REAL(In[7]))
				              +   0.039156*(REAL(In[-6])+REAL(In[6]))
				              +   0.0435387*(REAL(In[-5])+REAL(In[5]))
				              +   0.047487*(REAL(In[-4])+REAL(In[4]))
				              +   0.0508038*(REAL(In[-3])+REAL(In[3]))
				              +   0.0533139*(REAL(In[-2])+REAL(In[2]))
				              +   0.0548791*(REAL(In[-1])+REAL(In[1]))
				              +   0.0554109*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num76(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-29),-29,29,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num77 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num77(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.1756e-05*(REAL(In[-30])+REAL(In[30]))
				              +   2.04491e-05*(REAL(In[-29])+REAL(In[29]))
				              +   3.49092e-05*(REAL(In[-28])+REAL(In[28]))
				              +   5.84864e-05*(REAL(In[-27])+REAL(In[27]))
				              +   9.61657e-05*(REAL(In[-26])+REAL(In[26]))
				              +   0.00015518*(REAL(In[-25])+REAL(In[25]))
				              +   0.000245755*(REAL(In[-24])+REAL(In[24]))
				              +   0.000381961*(REAL(In[-23])+REAL(In[23]))
				              +   0.000582621*(REAL(In[-22])+REAL(In[22]))
				              +   0.000872175*(REAL(In[-21])+REAL(In[21]))
				              +   0.00128136*(REAL(In[-20])+REAL(In[20]))
				              +   0.00184752*(REAL(In[-19])+REAL(In[19]))
				              +   0.00261431*(REAL(In[-18])+REAL(In[18]))
				              +   0.00363058*(REAL(In[-17])+REAL(In[17]))
				              +   0.00494818*(REAL(In[-16])+REAL(In[16]))
				              +   0.00661858*(REAL(In[-15])+REAL(In[15]))
				              +   0.0086883*(REAL(In[-14])+REAL(In[14]))
				              +   0.0111932*(REAL(In[-13])+REAL(In[13]))
				              +   0.0141522*(REAL(In[-12])+REAL(In[12]))
				              +   0.0175609*(REAL(In[-11])+REAL(In[11]))
				              +   0.0213854*(REAL(In[-10])+REAL(In[10]))
				              +   0.0255587*(REAL(In[-9])+REAL(In[9]))
				              +   0.0299786*(REAL(In[-8])+REAL(In[8]))
				              +   0.0345091*(REAL(In[-7])+REAL(In[7]))
				              +   0.0389858*(REAL(In[-6])+REAL(In[6]))
				              +   0.0432244*(REAL(In[-5])+REAL(In[5]))
				              +   0.047033*(REAL(In[-4])+REAL(In[4]))
				              +   0.0502257*(REAL(In[-3])+REAL(In[3]))
				              +   0.0526381*(REAL(In[-2])+REAL(In[2]))
				              +   0.0541408*(REAL(In[-1])+REAL(In[1]))
				              +   0.0546512*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num77(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-30),-30,30,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num78 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num78(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.4548e-05*(REAL(In[-30])+REAL(In[30]))
				              +   2.49324e-05*(REAL(In[-29])+REAL(In[29]))
				              +   4.19559e-05*(REAL(In[-28])+REAL(In[28]))
				              +   6.93254e-05*(REAL(In[-27])+REAL(In[27]))
				              +   0.000112476*(REAL(In[-26])+REAL(In[26]))
				              +   0.000179183*(REAL(In[-25])+REAL(In[25]))
				              +   0.000280287*(REAL(In[-24])+REAL(In[24]))
				              +   0.000430506*(REAL(In[-23])+REAL(In[23]))
				              +   0.000649268*(REAL(In[-22])+REAL(In[22]))
				              +   0.000961474*(REAL(In[-21])+REAL(In[21]))
				              +   0.00139804*(REAL(In[-20])+REAL(In[20]))
				              +   0.00199605*(REAL(In[-19])+REAL(In[19]))
				              +   0.00279829*(REAL(In[-18])+REAL(In[18]))
				              +   0.00385198*(REAL(In[-17])+REAL(In[17]))
				              +   0.00520647*(REAL(In[-16])+REAL(In[16]))
				              +   0.0069099*(REAL(In[-15])+REAL(In[15]))
				              +   0.0090047*(REAL(In[-14])+REAL(In[14]))
				              +   0.0115222*(REAL(In[-13])+REAL(In[13]))
				              +   0.0144768*(REAL(In[-12])+REAL(In[12]))
				              +   0.0178598*(REAL(In[-11])+REAL(In[11]))
				              +   0.0216347*(REAL(In[-10])+REAL(In[10]))
				              +   0.0257332*(REAL(In[-9])+REAL(In[9]))
				              +   0.0300543*(REAL(In[-8])+REAL(In[8]))
				              +   0.0344658*(REAL(In[-7])+REAL(In[7]))
				              +   0.0388096*(REAL(In[-6])+REAL(In[6]))
				              +   0.04291*(REAL(In[-5])+REAL(In[5]))
				              +   0.0465852*(REAL(In[-4])+REAL(In[4]))
				              +   0.0496599*(REAL(In[-3])+REAL(In[3]))
				              +   0.0519796*(REAL(In[-2])+REAL(In[2]))
				              +   0.0534231*(REAL(In[-1])+REAL(In[1]))
				              +   0.0539131*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num78(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-30),-30,30,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num79 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num79(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.78449e-05*(REAL(In[-30])+REAL(In[30]))
				              +   3.01493e-05*(REAL(In[-29])+REAL(In[29]))
				              +   5.00402e-05*(REAL(In[-28])+REAL(In[28]))
				              +   8.15908e-05*(REAL(In[-27])+REAL(In[27]))
				              +   0.00013069*(REAL(In[-26])+REAL(In[26]))
				              +   0.000205647*(REAL(In[-25])+REAL(In[25]))
				              +   0.000317893*(REAL(In[-24])+REAL(In[24]))
				              +   0.000482748*(REAL(In[-23])+REAL(In[23]))
				              +   0.000720175*(REAL(In[-22])+REAL(In[22]))
				              +   0.00105544*(REAL(In[-21])+REAL(In[21]))
				              +   0.00151954*(REAL(In[-20])+REAL(In[20]))
				              +   0.00214915*(REAL(In[-19])+REAL(In[19]))
				              +   0.00298608*(REAL(In[-18])+REAL(In[18]))
				              +   0.00407582*(REAL(In[-17])+REAL(In[17]))
				              +   0.00546522*(REAL(In[-16])+REAL(In[16]))
				              +   0.00719913*(REAL(In[-15])+REAL(In[15]))
				              +   0.00931605*(REAL(In[-14])+REAL(In[14]))
				              +   0.011843*(REAL(In[-13])+REAL(In[13]))
				              +   0.0147901*(REAL(In[-12])+REAL(In[12]))
				              +   0.0181452*(REAL(In[-11])+REAL(In[11]))
				              +   0.021869*(REAL(In[-10])+REAL(In[10]))
				              +   0.0258927*(REAL(In[-9])+REAL(In[9]))
				              +   0.0301165*(REAL(In[-8])+REAL(In[8]))
				              +   0.034412*(REAL(In[-7])+REAL(In[7]))
				              +   0.0386274*(REAL(In[-6])+REAL(In[6]))
				              +   0.0425951*(REAL(In[-5])+REAL(In[5]))
				              +   0.0461427*(REAL(In[-4])+REAL(In[4]))
				              +   0.049105*(REAL(In[-3])+REAL(In[3]))
				              +   0.0513367*(REAL(In[-2])+REAL(In[2]))
				              +   0.0527241*(REAL(In[-1])+REAL(In[1]))
				              +   0.0531948*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num79(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-30),-30,30,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num80 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num80(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.28014e-05*(REAL(In[-31])+REAL(In[31]))
				              +   2.17062e-05*(REAL(In[-30])+REAL(In[30]))
				              +   3.61737e-05*(REAL(In[-29])+REAL(In[29]))
				              +   5.92493e-05*(REAL(In[-28])+REAL(In[28]))
				              +   9.53793e-05*(REAL(In[-27])+REAL(In[27]))
				              +   0.000150906*(REAL(In[-26])+REAL(In[26]))
				              +   0.00023466*(REAL(In[-25])+REAL(In[25]))
				              +   0.000358636*(REAL(In[-24])+REAL(In[24]))
				              +   0.000538703*(REAL(In[-23])+REAL(In[23]))
				              +   0.00079529*(REAL(In[-22])+REAL(In[22]))
				              +   0.00115394*(REAL(In[-21])+REAL(In[21]))
				              +   0.00164559*(REAL(In[-20])+REAL(In[20]))
				              +   0.00230643*(REAL(In[-19])+REAL(In[19]))
				              +   0.00317718*(REAL(In[-18])+REAL(In[18]))
				              +   0.00430153*(REAL(In[-17])+REAL(In[17]))
				              +   0.00572382*(REAL(In[-16])+REAL(In[16]))
				              +   0.00748565*(REAL(In[-15])+REAL(In[15]))
				              +   0.00962175*(REAL(In[-14])+REAL(In[14]))
				              +   0.0121551*(REAL(In[-13])+REAL(In[13]))
				              +   0.015092*(REAL(In[-12])+REAL(In[12]))
				              +   0.0184168*(REAL(In[-11])+REAL(In[11]))
				              +   0.0220884*(REAL(In[-10])+REAL(In[10]))
				              +   0.0260372*(REAL(In[-9])+REAL(In[9]))
				              +   0.0301651*(REAL(In[-8])+REAL(In[8]))
				              +   0.0343476*(REAL(In[-7])+REAL(In[7]))
				              +   0.0384388*(REAL(In[-6])+REAL(In[6]))
				              +   0.042279*(REAL(In[-5])+REAL(In[5]))
				              +   0.0457046*(REAL(In[-4])+REAL(In[4]))
				              +   0.0485597*(REAL(In[-3])+REAL(In[3]))
				              +   0.0507076*(REAL(In[-2])+REAL(In[2]))
				              +   0.0520417*(REAL(In[-1])+REAL(In[1]))
				              +   0.0524942*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num80(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-31),-31,31,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num81 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num81(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.56608e-05*(REAL(In[-31])+REAL(In[31]))
				              +   2.61953e-05*(REAL(In[-30])+REAL(In[30]))
				              +   4.30833e-05*(REAL(In[-29])+REAL(In[29]))
				              +   6.96738e-05*(REAL(In[-28])+REAL(In[28]))
				              +   0.000110791*(REAL(In[-27])+REAL(In[27]))
				              +   0.000173227*(REAL(In[-26])+REAL(In[26]))
				              +   0.000266318*(REAL(In[-25])+REAL(In[25]))
				              +   0.000402588*(REAL(In[-24])+REAL(In[24]))
				              +   0.000598407*(REAL(In[-23])+REAL(In[23]))
				              +   0.000874595*(REAL(In[-22])+REAL(In[22]))
				              +   0.00125688*(REAL(In[-21])+REAL(In[21]))
				              +   0.00177604*(REAL(In[-20])+REAL(In[20]))
				              +   0.00246769*(REAL(In[-19])+REAL(In[19]))
				              +   0.00337133*(REAL(In[-18])+REAL(In[18]))
				              +   0.00452884*(REAL(In[-17])+REAL(In[17]))
				              +   0.00598202*(REAL(In[-16])+REAL(In[16]))
				              +   0.00776934*(REAL(In[-15])+REAL(In[15]))
				              +   0.00992192*(REAL(In[-14])+REAL(In[14]))
				              +   0.012459*(REAL(In[-13])+REAL(In[13]))
				              +   0.0153831*(REAL(In[-12])+REAL(In[12]))
				              +   0.0186758*(REAL(In[-11])+REAL(In[11]))
				              +   0.0222942*(REAL(In[-10])+REAL(In[10]))
				              +   0.0261685*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302023*(REAL(In[-8])+REAL(In[8]))
				              +   0.034275*(REAL(In[-7])+REAL(In[7]))
				              +   0.0382463*(REAL(In[-6])+REAL(In[6]))
				              +   0.0419639*(REAL(In[-5])+REAL(In[5]))
				              +   0.0452729*(REAL(In[-4])+REAL(In[4]))
				              +   0.0480259*(REAL(In[-3])+REAL(In[3]))
				              +   0.0500943*(REAL(In[-2])+REAL(In[2]))
				              +   0.0513778*(REAL(In[-1])+REAL(In[1]))
				              +   0.0518129*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num81(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-31),-31,31,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num82 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num82(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.13244e-05*(REAL(In[-32])+REAL(In[32]))
				              +   1.90052e-05*(REAL(In[-31])+REAL(In[31]))
				              +   3.13756e-05*(REAL(In[-30])+REAL(In[30]))
				              +   5.09532e-05*(REAL(In[-29])+REAL(In[29]))
				              +   8.13979e-05*(REAL(In[-28])+REAL(In[28]))
				              +   0.000127914*(REAL(In[-27])+REAL(In[27]))
				              +   0.000197734*(REAL(In[-26])+REAL(In[26]))
				              +   0.000300683*(REAL(In[-25])+REAL(In[25]))
				              +   0.000449777*(REAL(In[-24])+REAL(In[24]))
				              +   0.000661831*(REAL(In[-23])+REAL(In[23]))
				              +   0.000957985*(REAL(In[-22])+REAL(In[22]))
				              +   0.00136406*(REAL(In[-21])+REAL(In[21]))
				              +   0.00191059*(REAL(In[-20])+REAL(In[20]))
				              +   0.00263248*(REAL(In[-19])+REAL(In[19]))
				              +   0.00356799*(REAL(In[-18])+REAL(In[18]))
				              +   0.00475711*(REAL(In[-17])+REAL(In[17]))
				              +   0.00623915*(REAL(In[-16])+REAL(In[16]))
				              +   0.00804951*(REAL(In[-15])+REAL(In[15]))
				              +   0.0102159*(REAL(In[-14])+REAL(In[14]))
				              +   0.0127539*(REAL(In[-13])+REAL(In[13]))
				              +   0.0156628*(REAL(In[-12])+REAL(In[12]))
				              +   0.0189217*(REAL(In[-11])+REAL(In[11]))
				              +   0.0224861*(REAL(In[-10])+REAL(In[10]))
				              +   0.0262862*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302276*(REAL(In[-8])+REAL(In[8]))
				              +   0.0341932*(REAL(In[-7])+REAL(In[7]))
				              +   0.0380486*(REAL(In[-6])+REAL(In[6]))
				              +   0.0416486*(REAL(In[-5])+REAL(In[5]))
				              +   0.0448459*(REAL(In[-4])+REAL(In[4]))
				              +   0.0475014*(REAL(In[-3])+REAL(In[3]))
				              +   0.0494939*(REAL(In[-2])+REAL(In[2]))
				              +   0.0507294*(REAL(In[-1])+REAL(In[1]))
				              +   0.051148*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num82(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-32),-32,32,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num83 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num83(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.38175e-05*(REAL(In[-32])+REAL(In[32]))
				              +   2.28891e-05*(REAL(In[-31])+REAL(In[31]))
				              +   3.73139e-05*(REAL(In[-30])+REAL(In[30]))
				              +   5.98622e-05*(REAL(In[-29])+REAL(In[29]))
				              +   9.45097e-05*(REAL(In[-28])+REAL(In[28]))
				              +   0.000146839*(REAL(In[-27])+REAL(In[27]))
				              +   0.000224516*(REAL(In[-26])+REAL(In[26]))
				              +   0.000337828*(REAL(In[-25])+REAL(In[25]))
				              +   0.000500247*(REAL(In[-24])+REAL(In[24]))
				              +   0.000728979*(REAL(In[-23])+REAL(In[23]))
				              +   0.00104541*(REAL(In[-22])+REAL(In[22]))
				              +   0.00147536*(REAL(In[-21])+REAL(In[21]))
				              +   0.00204905*(REAL(In[-20])+REAL(In[20]))
				              +   0.00280058*(REAL(In[-19])+REAL(In[19]))
				              +   0.00376691*(REAL(In[-18])+REAL(In[18]))
				              +   0.00498612*(REAL(In[-17])+REAL(In[17]))
				              +   0.00649504*(REAL(In[-16])+REAL(In[16]))
				              +   0.00832611*(REAL(In[-15])+REAL(In[15]))
				              +   0.0105037*(REAL(In[-14])+REAL(In[14]))
				              +   0.0130403*(REAL(In[-13])+REAL(In[13]))
				              +   0.015932*(REAL(In[-12])+REAL(In[12]))
				              +   0.0191556*(REAL(In[-11])+REAL(In[11]))
				              +   0.0226654*(REAL(In[-10])+REAL(In[10]))
				              +   0.0263919*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302427*(REAL(In[-8])+REAL(In[8]))
				              +   0.0341044*(REAL(In[-7])+REAL(In[7]))
				              +   0.037848*(REAL(In[-6])+REAL(In[6]))
				              +   0.0413348*(REAL(In[-5])+REAL(In[5]))
				              +   0.0444252*(REAL(In[-4])+REAL(In[4]))
				              +   0.0469878*(REAL(In[-3])+REAL(In[3]))
				              +   0.0489083*(REAL(In[-2])+REAL(In[2]))
				              +   0.050098*(REAL(In[-1])+REAL(In[1]))
				              +   0.050501*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num83(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-32),-32,32,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num84 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num84(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.67296e-05*(REAL(In[-32])+REAL(In[32]))
				              +   2.73678e-05*(REAL(In[-31])+REAL(In[31]))
				              +   4.40766e-05*(REAL(In[-30])+REAL(In[30]))
				              +   6.9886e-05*(REAL(In[-29])+REAL(In[29]))
				              +   0.000109091*(REAL(In[-28])+REAL(In[28]))
				              +   0.000167648*(REAL(In[-27])+REAL(In[27]))
				              +   0.000253643*(REAL(In[-26])+REAL(In[26]))
				              +   0.0003778*(REAL(In[-25])+REAL(In[25]))
				              +   0.000554008*(REAL(In[-24])+REAL(In[24]))
				              +   0.000799803*(REAL(In[-23])+REAL(In[23]))
				              +   0.00113675*(REAL(In[-22])+REAL(In[22]))
				              +   0.0015906*(REAL(In[-21])+REAL(In[21]))
				              +   0.00219114*(REAL(In[-20])+REAL(In[20]))
				              +   0.00297163*(REAL(In[-19])+REAL(In[19]))
				              +   0.00396765*(REAL(In[-18])+REAL(In[18]))
				              +   0.00521537*(REAL(In[-17])+REAL(In[17]))
				              +   0.00674919*(REAL(In[-16])+REAL(In[16]))
				              +   0.0085987*(REAL(In[-15])+REAL(In[15]))
				              +   0.0107852*(REAL(In[-14])+REAL(In[14]))
				              +   0.0133179*(REAL(In[-13])+REAL(In[13]))
				              +   0.0161905*(REAL(In[-12])+REAL(In[12]))
				              +   0.0193775*(REAL(In[-11])+REAL(In[11]))
				              +   0.0228322*(REAL(In[-10])+REAL(In[10]))
				              +   0.0264859*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302478*(REAL(In[-8])+REAL(In[8]))
				              +   0.0340085*(REAL(In[-7])+REAL(In[7]))
				              +   0.037644*(REAL(In[-6])+REAL(In[6]))
				              +   0.0410221*(REAL(In[-5])+REAL(In[5]))
				              +   0.0440103*(REAL(In[-4])+REAL(In[4]))
				              +   0.0464841*(REAL(In[-3])+REAL(In[3]))
				              +   0.0483358*(REAL(In[-2])+REAL(In[2]))
				              +   0.0494821*(REAL(In[-1])+REAL(In[1]))
				              +   0.0498702*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num84(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-32),-32,32,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num85 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num85(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.22522e-05*(REAL(In[-33])+REAL(In[33]))
				              +   2.01067e-05*(REAL(In[-32])+REAL(In[32]))
				              +   3.24975e-05*(REAL(In[-31])+REAL(In[31]))
				              +   5.17297e-05*(REAL(In[-30])+REAL(In[30]))
				              +   8.10981e-05*(REAL(In[-29])+REAL(In[29]))
				              +   0.000125216*(REAL(In[-28])+REAL(In[28]))
				              +   0.000190411*(REAL(In[-27])+REAL(In[27]))
				              +   0.000285171*(REAL(In[-26])+REAL(In[26]))
				              +   0.000420628*(REAL(In[-25])+REAL(In[25]))
				              +   0.000611044*(REAL(In[-24])+REAL(In[24]))
				              +   0.000874232*(REAL(In[-23])+REAL(In[23]))
				              +   0.00123186*(REAL(In[-22])+REAL(In[22]))
				              +   0.00170953*(REAL(In[-21])+REAL(In[21]))
				              +   0.00233654*(REAL(In[-20])+REAL(In[20]))
				              +   0.00314522*(REAL(In[-19])+REAL(In[19]))
				              +   0.00416973*(REAL(In[-18])+REAL(In[18]))
				              +   0.00544436*(REAL(In[-17])+REAL(In[17]))
				              +   0.00700109*(REAL(In[-16])+REAL(In[16]))
				              +   0.00886676*(REAL(In[-15])+REAL(In[15]))
				              +   0.0110598*(REAL(In[-14])+REAL(In[14]))
				              +   0.0135865*(REAL(In[-13])+REAL(In[13]))
				              +   0.016438*(REAL(In[-12])+REAL(In[12]))
				              +   0.0195871*(REAL(In[-11])+REAL(In[11]))
				              +   0.0229865*(REAL(In[-10])+REAL(In[10]))
				              +   0.0265679*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302428*(REAL(In[-8])+REAL(In[8]))
				              +   0.0339052*(REAL(In[-7])+REAL(In[7]))
				              +   0.0374362*(REAL(In[-6])+REAL(In[6]))
				              +   0.0407098*(REAL(In[-5])+REAL(In[5]))
				              +   0.0435999*(REAL(In[-4])+REAL(In[4]))
				              +   0.0459889*(REAL(In[-3])+REAL(In[3]))
				              +   0.0477751*(REAL(In[-2])+REAL(In[2]))
				              +   0.0488799*(REAL(In[-1])+REAL(In[1]))
				              +   0.0492539*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num85(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-33),-33,33,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num86 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num86(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.47999e-05*(REAL(In[-33])+REAL(In[33]))
				              +   2.39979e-05*(REAL(In[-32])+REAL(In[32]))
				              +   3.83378e-05*(REAL(In[-31])+REAL(In[31]))
				              +   6.03425e-05*(REAL(In[-30])+REAL(In[30]))
				              +   9.3575e-05*(REAL(In[-29])+REAL(In[29]))
				              +   0.000142968*(REAL(In[-28])+REAL(In[28]))
				              +   0.000215207*(REAL(In[-27])+REAL(In[27]))
				              +   0.000319166*(REAL(In[-26])+REAL(In[26]))
				              +   0.000466356*(REAL(In[-25])+REAL(In[25]))
				              +   0.000671367*(REAL(In[-24])+REAL(In[24]))
				              +   0.000952234*(REAL(In[-23])+REAL(In[23]))
				              +   0.00133066*(REAL(In[-22])+REAL(In[22]))
				              +   0.00183204*(REAL(In[-21])+REAL(In[21]))
				              +   0.00248508*(REAL(In[-20])+REAL(In[20]))
				              +   0.00332115*(REAL(In[-19])+REAL(In[19]))
				              +   0.00437299*(REAL(In[-18])+REAL(In[18]))
				              +   0.00567294*(REAL(In[-17])+REAL(In[17]))
				              +   0.0072507*(REAL(In[-16])+REAL(In[16]))
				              +   0.00913045*(REAL(In[-15])+REAL(In[15]))
				              +   0.0113278*(REAL(In[-14])+REAL(In[14]))
				              +   0.0138465*(REAL(In[-13])+REAL(In[13]))
				              +   0.0166754*(REAL(In[-12])+REAL(In[12]))
				              +   0.0197858*(REAL(In[-11])+REAL(In[11]))
				              +   0.0231298*(REAL(In[-10])+REAL(In[10]))
				              +   0.0266398*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302295*(REAL(In[-8])+REAL(In[8]))
				              +   0.0337966*(REAL(In[-7])+REAL(In[7]))
				              +   0.0372268*(REAL(In[-6])+REAL(In[6]))
				              +   0.0403998*(REAL(In[-5])+REAL(In[5]))
				              +   0.0431961*(REAL(In[-4])+REAL(In[4]))
				              +   0.0455041*(REAL(In[-3])+REAL(In[3]))
				              +   0.0472278*(REAL(In[-2])+REAL(In[2]))
				              +   0.0482932*(REAL(In[-1])+REAL(In[1]))
				              +   0.0486536*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num86(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-33),-33,33,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num87 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num87(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.09153e-05*(REAL(In[-34])+REAL(In[34]))
				              +   1.7751e-05*(REAL(In[-33])+REAL(In[33]))
				              +   2.84518e-05*(REAL(In[-32])+REAL(In[32]))
				              +   4.4946e-05*(REAL(In[-31])+REAL(In[31]))
				              +   6.9979e-05*(REAL(In[-30])+REAL(In[30]))
				              +   0.000107384*(REAL(In[-29])+REAL(In[29]))
				              +   0.000162409*(REAL(In[-28])+REAL(In[28]))
				              +   0.000242089*(REAL(In[-27])+REAL(In[27]))
				              +   0.00035566*(REAL(In[-26])+REAL(In[26]))
				              +   0.000514982*(REAL(In[-25])+REAL(In[25]))
				              +   0.000734927*(REAL(In[-24])+REAL(In[24]))
				              +   0.00103369*(REAL(In[-23])+REAL(In[23]))
				              +   0.00143297*(REAL(In[-22])+REAL(In[22]))
				              +   0.00195783*(REAL(In[-21])+REAL(In[21]))
				              +   0.0026364*(REAL(In[-20])+REAL(In[20]))
				              +   0.00349899*(REAL(In[-19])+REAL(In[19]))
				              +   0.00457688*(REAL(In[-18])+REAL(In[18]))
				              +   0.00590055*(REAL(In[-17])+REAL(In[17]))
				              +   0.00749742*(REAL(In[-16])+REAL(In[16]))
				              +   0.00938915*(REAL(In[-15])+REAL(In[15]))
				              +   0.0115888*(REAL(In[-14])+REAL(In[14]))
				              +   0.0140975*(REAL(In[-13])+REAL(In[13]))
				              +   0.0169023*(REAL(In[-12])+REAL(In[12]))
				              +   0.019973*(REAL(In[-11])+REAL(In[11]))
				              +   0.0232615*(REAL(In[-10])+REAL(In[10]))
				              +   0.026701*(REAL(In[-9])+REAL(In[9]))
				              +   0.0302073*(REAL(In[-8])+REAL(In[8]))
				              +   0.0336817*(REAL(In[-7])+REAL(In[7]))
				              +   0.0370144*(REAL(In[-6])+REAL(In[6]))
				              +   0.0400907*(REAL(In[-5])+REAL(In[5]))
				              +   0.0427969*(REAL(In[-4])+REAL(In[4]))
				              +   0.0450274*(REAL(In[-3])+REAL(In[3]))
				              +   0.0466915*(REAL(In[-2])+REAL(In[2]))
				              +   0.0477193*(REAL(In[-1])+REAL(In[1]))
				              +   0.0480669*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num87(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-34),-34,34,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num88 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num88(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.31546e-05*(REAL(In[-34])+REAL(In[34]))
				              +   2.1148e-05*(REAL(In[-33])+REAL(In[33]))
				              +   3.35201e-05*(REAL(In[-32])+REAL(In[32]))
				              +   5.23825e-05*(REAL(In[-31])+REAL(In[31]))
				              +   8.07073e-05*(REAL(In[-30])+REAL(In[30]))
				              +   0.000122598*(REAL(In[-29])+REAL(In[29]))
				              +   0.000183612*(REAL(In[-28])+REAL(In[28]))
				              +   0.000271121*(REAL(In[-27])+REAL(In[27]))
				              +   0.000394702*(REAL(In[-26])+REAL(In[26]))
				              +   0.000566527*(REAL(In[-25])+REAL(In[25]))
				              +   0.00080171*(REAL(In[-24])+REAL(In[24]))
				              +   0.00111856*(REAL(In[-23])+REAL(In[23]))
				              +   0.00153867*(REAL(In[-22])+REAL(In[22]))
				              +   0.00208679*(REAL(In[-21])+REAL(In[21]))
				              +   0.00279032*(REAL(In[-20])+REAL(In[20]))
				              +   0.00367855*(REAL(In[-19])+REAL(In[19]))
				              +   0.00478127*(REAL(In[-18])+REAL(In[18]))
				              +   0.0061271*(REAL(In[-17])+REAL(In[17]))
				              +   0.00774127*(REAL(In[-16])+REAL(In[16]))
				              +   0.00964304*(REAL(In[-15])+REAL(In[15]))
				              +   0.011843*(REAL(In[-14])+REAL(In[14]))
				              +   0.0143401*(REAL(In[-13])+REAL(In[13]))
				              +   0.0171195*(REAL(In[-12])+REAL(In[12]))
				              +   0.0201499*(REAL(In[-11])+REAL(In[11]))
				              +   0.023383*(REAL(In[-10])+REAL(In[10]))
				              +   0.026753*(REAL(In[-9])+REAL(In[9]))
				              +   0.0301779*(REAL(In[-8])+REAL(In[8]))
				              +   0.0335623*(REAL(In[-7])+REAL(In[7]))
				              +   0.0368009*(REAL(In[-6])+REAL(In[6]))
				              +   0.0397842*(REAL(In[-5])+REAL(In[5]))
				              +   0.0424041*(REAL(In[-4])+REAL(In[4]))
				              +   0.0445606*(REAL(In[-3])+REAL(In[3]))
				              +   0.0461677*(REAL(In[-2])+REAL(In[2]))
				              +   0.0471596*(REAL(In[-1])+REAL(In[1]))
				              +   0.047495*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num88(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-34),-34,34,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num89 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num89(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.57455e-05*(REAL(In[-34])+REAL(In[34]))
				              +   2.50337e-05*(REAL(In[-33])+REAL(In[33]))
				              +   3.92538e-05*(REAL(In[-32])+REAL(In[32]))
				              +   6.07056e-05*(REAL(In[-31])+REAL(In[31]))
				              +   9.259e-05*(REAL(In[-30])+REAL(In[30]))
				              +   0.00013928*(REAL(In[-29])+REAL(In[29]))
				              +   0.000206634*(REAL(In[-28])+REAL(In[28]))
				              +   0.000302346*(REAL(In[-27])+REAL(In[27]))
				              +   0.00043631*(REAL(In[-26])+REAL(In[26]))
				              +   0.000620978*(REAL(In[-25])+REAL(In[25]))
				              +   0.000871657*(REAL(In[-24])+REAL(In[24]))
				              +   0.00120671*(REAL(In[-23])+REAL(In[23]))
				              +   0.0016476*(REAL(In[-22])+REAL(In[22]))
				              +   0.00221864*(REAL(In[-21])+REAL(In[21]))
				              +   0.00294655*(REAL(In[-20])+REAL(In[20]))
				              +   0.00385947*(REAL(In[-19])+REAL(In[19]))
				              +   0.00498576*(REAL(In[-18])+REAL(In[18]))
				              +   0.00635219*(REAL(In[-17])+REAL(In[17]))
				              +   0.00798187*(REAL(In[-16])+REAL(In[16]))
				              +   0.0098918*(REAL(In[-15])+REAL(In[15]))
				              +   0.0120902*(REAL(In[-14])+REAL(In[14]))
				              +   0.0145741*(REAL(In[-13])+REAL(In[13]))
				              +   0.0173269*(REAL(In[-12])+REAL(In[12]))
				              +   0.0203164*(REAL(In[-11])+REAL(In[11]))
				              +   0.0234943*(REAL(In[-10])+REAL(In[10]))
				              +   0.0267958*(REAL(In[-9])+REAL(In[9]))
				              +   0.0301412*(REAL(In[-8])+REAL(In[8]))
				              +   0.0334382*(REAL(In[-7])+REAL(In[7]))
				              +   0.036586*(REAL(In[-6])+REAL(In[6]))
				              +   0.0394798*(REAL(In[-5])+REAL(In[5]))
				              +   0.0420169*(REAL(In[-4])+REAL(In[4]))
				              +   0.0441025*(REAL(In[-3])+REAL(In[3]))
				              +   0.0456552*(REAL(In[-2])+REAL(In[2]))
				              +   0.046613*(REAL(In[-1])+REAL(In[1]))
				              +   0.0469367*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num89(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-34),-34,34,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num90 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num90(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.1744e-05*(REAL(In[-35])+REAL(In[35]))
				              +   1.87241e-05*(REAL(In[-34])+REAL(In[34]))
				              +   2.94519e-05*(REAL(In[-33])+REAL(In[33]))
				              +   4.5704e-05*(REAL(In[-32])+REAL(In[32]))
				              +   6.99717e-05*(REAL(In[-31])+REAL(In[31]))
				              +   0.000105686*(REAL(In[-30])+REAL(In[30]))
				              +   0.000157487*(REAL(In[-29])+REAL(In[29]))
				              +   0.000231524*(REAL(In[-28])+REAL(In[28]))
				              +   0.000335797*(REAL(In[-27])+REAL(In[27]))
				              +   0.000480491*(REAL(In[-26])+REAL(In[26]))
				              +   0.0006783*(REAL(In[-25])+REAL(In[25]))
				              +   0.000944683*(REAL(In[-24])+REAL(In[24]))
				              +   0.00129801*(REAL(In[-23])+REAL(In[23]))
				              +   0.00175954*(REAL(In[-22])+REAL(In[22]))
				              +   0.00235314*(REAL(In[-21])+REAL(In[21]))
				              +   0.00310473*(REAL(In[-20])+REAL(In[20]))
				              +   0.00404136*(REAL(In[-19])+REAL(In[19]))
				              +   0.00518992*(REAL(In[-18])+REAL(In[18]))
				              +   0.00657538*(REAL(In[-17])+REAL(In[17]))
				              +   0.00821881*(REAL(In[-16])+REAL(In[16]))
				              +   0.010135*(REAL(In[-15])+REAL(In[15]))
				              +   0.0123302*(REAL(In[-14])+REAL(In[14]))
				              +   0.0147993*(REAL(In[-13])+REAL(In[13]))
				              +   0.0175244*(REAL(In[-12])+REAL(In[12]))
				              +   0.0204725*(REAL(In[-11])+REAL(In[11]))
				              +   0.0235954*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268294*(REAL(In[-9])+REAL(In[9]))
				              +   0.030097*(REAL(In[-8])+REAL(In[8]))
				              +   0.0333091*(REAL(In[-7])+REAL(In[7]))
				              +   0.036369*(REAL(In[-6])+REAL(In[6]))
				              +   0.0391766*(REAL(In[-5])+REAL(In[5]))
				              +   0.0416343*(REAL(In[-4])+REAL(In[4]))
				              +   0.0436519*(REAL(In[-3])+REAL(In[3]))
				              +   0.0451527*(REAL(In[-2])+REAL(In[2]))
				              +   0.0460778*(REAL(In[-1])+REAL(In[1]))
				              +   0.0463903*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num90(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-35),-35,35,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num91 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num91(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.40283e-05*(REAL(In[-35])+REAL(In[35]))
				              +   2.21288e-05*(REAL(In[-34])+REAL(In[34]))
				              +   3.44489e-05*(REAL(In[-33])+REAL(In[33]))
				              +   5.29241e-05*(REAL(In[-32])+REAL(In[32]))
				              +   8.02407e-05*(REAL(In[-31])+REAL(In[31]))
				              +   0.00012006*(REAL(In[-30])+REAL(In[30]))
				              +   0.000177281*(REAL(In[-29])+REAL(In[29]))
				              +   0.000258339*(REAL(In[-28])+REAL(In[28]))
				              +   0.000371518*(REAL(In[-27])+REAL(In[27]))
				              +   0.000527269*(REAL(In[-26])+REAL(In[26]))
				              +   0.000738492*(REAL(In[-25])+REAL(In[25]))
				              +   0.00102076*(REAL(In[-24])+REAL(In[24]))
				              +   0.00139239*(REAL(In[-23])+REAL(In[23]))
				              +   0.0018744*(REAL(In[-22])+REAL(In[22]))
				              +   0.00249014*(REAL(In[-21])+REAL(In[21]))
				              +   0.00326474*(REAL(In[-20])+REAL(In[20]))
				              +   0.00422412*(REAL(In[-19])+REAL(In[19]))
				              +   0.00539369*(REAL(In[-18])+REAL(In[18]))
				              +   0.00679669*(REAL(In[-17])+REAL(In[17]))
				              +   0.00845223*(REAL(In[-16])+REAL(In[16]))
				              +   0.0103731*(REAL(In[-15])+REAL(In[15]))
				              +   0.0125633*(REAL(In[-14])+REAL(In[14]))
				              +   0.0150164*(REAL(In[-13])+REAL(In[13]))
				              +   0.0177128*(REAL(In[-12])+REAL(In[12]))
				              +   0.0206193*(REAL(In[-11])+REAL(In[11]))
				              +   0.0236875*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268552*(REAL(In[-9])+REAL(In[9]))
				              +   0.0300469*(REAL(In[-8])+REAL(In[8]))
				              +   0.0331768*(REAL(In[-7])+REAL(In[7]))
				              +   0.0361518*(REAL(In[-6])+REAL(In[6]))
				              +   0.0388765*(REAL(In[-5])+REAL(In[5]))
				              +   0.041258*(REAL(In[-4])+REAL(In[4]))
				              +   0.0432106*(REAL(In[-3])+REAL(In[3]))
				              +   0.0446616*(REAL(In[-2])+REAL(In[2]))
				              +   0.0455555*(REAL(In[-1])+REAL(In[1]))
				              +   0.0458575*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num91(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-35),-35,35,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num92 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num92(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.05285e-05*(REAL(In[-36])+REAL(In[36]))
				              +   1.66516e-05*(REAL(In[-35])+REAL(In[35]))
				              +   2.59978e-05*(REAL(In[-34])+REAL(In[34]))
				              +   4.0069e-05*(REAL(In[-33])+REAL(In[33]))
				              +   6.0964e-05*(REAL(In[-32])+REAL(In[32]))
				              +   9.15651e-05*(REAL(In[-31])+REAL(In[31]))
				              +   0.000135762*(REAL(In[-30])+REAL(In[30]))
				              +   0.000198709*(REAL(In[-29])+REAL(In[29]))
				              +   0.000287111*(REAL(In[-28])+REAL(In[28]))
				              +   0.000409519*(REAL(In[-27])+REAL(In[27]))
				              +   0.00057662*(REAL(In[-26])+REAL(In[26]))
				              +   0.000801489*(REAL(In[-25])+REAL(In[25]))
				              +   0.00109976*(REAL(In[-24])+REAL(In[24]))
				              +   0.00148966*(REAL(In[-23])+REAL(In[23]))
				              +   0.00199192*(REAL(In[-22])+REAL(In[22]))
				              +   0.00262934*(REAL(In[-21])+REAL(In[21]))
				              +   0.00342621*(REAL(In[-20])+REAL(In[20]))
				              +   0.0044073*(REAL(In[-19])+REAL(In[19]))
				              +   0.00559658*(REAL(In[-18])+REAL(In[18]))
				              +   0.00701561*(REAL(In[-17])+REAL(In[17]))
				              +   0.0086816*(REAL(In[-16])+REAL(In[16]))
				              +   0.0106054*(REAL(In[-15])+REAL(In[15]))
				              +   0.0127892*(REAL(In[-14])+REAL(In[14]))
				              +   0.0152249*(REAL(In[-13])+REAL(In[13]))
				              +   0.0178919*(REAL(In[-12])+REAL(In[12]))
				              +   0.0207563*(REAL(In[-11])+REAL(In[11]))
				              +   0.0237703*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268727*(REAL(In[-9])+REAL(In[9]))
				              +   0.0299903*(REAL(In[-8])+REAL(In[8]))
				              +   0.0330402*(REAL(In[-7])+REAL(In[7]))
				              +   0.0359331*(REAL(In[-6])+REAL(In[6]))
				              +   0.038578*(REAL(In[-5])+REAL(In[5]))
				              +   0.0408862*(REAL(In[-4])+REAL(In[4]))
				              +   0.0427765*(REAL(In[-3])+REAL(In[3]))
				              +   0.04418*(REAL(In[-2])+REAL(In[2]))
				              +   0.0450441*(REAL(In[-1])+REAL(In[1]))
				              +   0.0453359*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num92(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-36),-36,36,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num93 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num93(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.25507e-05*(REAL(In[-36])+REAL(In[36]))
				              +   1.96475e-05*(REAL(In[-35])+REAL(In[35]))
				              +   3.03715e-05*(REAL(In[-34])+REAL(In[34]))
				              +   4.63596e-05*(REAL(In[-33])+REAL(In[33]))
				              +   6.98766e-05*(REAL(In[-32])+REAL(In[32]))
				              +   0.000104002*(REAL(In[-31])+REAL(In[31]))
				              +   0.000152851*(REAL(In[-30])+REAL(In[30]))
				              +   0.000221825*(REAL(In[-29])+REAL(In[29]))
				              +   0.000317886*(REAL(In[-28])+REAL(In[28]))
				              +   0.00044983*(REAL(In[-27])+REAL(In[27]))
				              +   0.000628555*(REAL(In[-26])+REAL(In[26]))
				              +   0.000867272*(REAL(In[-25])+REAL(In[25]))
				              +   0.00118164*(REAL(In[-24])+REAL(In[24]))
				              +   0.00158976*(REAL(In[-23])+REAL(In[23]))
				              +   0.002112*(REAL(In[-22])+REAL(In[22]))
				              +   0.00277061*(REAL(In[-21])+REAL(In[21]))
				              +   0.003589*(REAL(In[-20])+REAL(In[20]))
				              +   0.0045908*(REAL(In[-19])+REAL(In[19]))
				              +   0.00579856*(REAL(In[-18])+REAL(In[18]))
				              +   0.00723219*(REAL(In[-17])+REAL(In[17]))
				              +   0.0089071*(REAL(In[-16])+REAL(In[16]))
				              +   0.0108323*(REAL(In[-15])+REAL(In[15]))
				              +   0.0130083*(REAL(In[-14])+REAL(In[14]))
				              +   0.0154255*(REAL(In[-13])+REAL(In[13]))
				              +   0.0180624*(REAL(In[-12])+REAL(In[12]))
				              +   0.0208846*(REAL(In[-11])+REAL(In[11]))
				              +   0.023845*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268833*(REAL(In[-9])+REAL(In[9]))
				              +   0.0299287*(REAL(In[-8])+REAL(In[8]))
				              +   0.032901*(REAL(In[-7])+REAL(In[7]))
				              +   0.0357147*(REAL(In[-6])+REAL(In[6]))
				              +   0.0382827*(REAL(In[-5])+REAL(In[5]))
				              +   0.0405205*(REAL(In[-4])+REAL(In[4]))
				              +   0.0423511*(REAL(In[-3])+REAL(In[3]))
				              +   0.0437091*(REAL(In[-2])+REAL(In[2]))
				              +   0.0445447*(REAL(In[-1])+REAL(In[1]))
				              +   0.0448268*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num93(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-36),-36,36,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num94 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num94(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.48708e-05*(REAL(In[-36])+REAL(In[36]))
				              +   2.30501e-05*(REAL(In[-35])+REAL(In[35]))
				              +   3.52898e-05*(REAL(In[-34])+REAL(In[34]))
				              +   5.33661e-05*(REAL(In[-33])+REAL(In[33]))
				              +   7.97111e-05*(REAL(In[-32])+REAL(In[32]))
				              +   0.000117601*(REAL(In[-31])+REAL(In[31]))
				              +   0.000171373*(REAL(In[-30])+REAL(In[30]))
				              +   0.000246667*(REAL(In[-29])+REAL(In[29]))
				              +   0.000350686*(REAL(In[-28])+REAL(In[28]))
				              +   0.000492452*(REAL(In[-27])+REAL(In[27]))
				              +   0.000683043*(REAL(In[-26])+REAL(In[26]))
				              +   0.000935772*(REAL(In[-25])+REAL(In[25]))
				              +   0.00126628*(REAL(In[-24])+REAL(In[24]))
				              +   0.0016925*(REAL(In[-23])+REAL(In[23]))
				              +   0.00223443*(REAL(In[-22])+REAL(In[22]))
				              +   0.00291369*(REAL(In[-21])+REAL(In[21]))
				              +   0.00375281*(REAL(In[-20])+REAL(In[20]))
				              +   0.0047743*(REAL(In[-19])+REAL(In[19]))
				              +   0.00599929*(REAL(In[-18])+REAL(In[18]))
				              +   0.00744611*(REAL(In[-17])+REAL(In[17]))
				              +   0.00912844*(REAL(In[-16])+REAL(In[16]))
				              +   0.0110536*(REAL(In[-15])+REAL(In[15]))
				              +   0.0132205*(REAL(In[-14])+REAL(In[14]))
				              +   0.0156181*(REAL(In[-13])+REAL(In[13]))
				              +   0.0182242*(REAL(In[-12])+REAL(In[12]))
				              +   0.0210043*(REAL(In[-11])+REAL(In[11]))
				              +   0.0239115*(REAL(In[-10])+REAL(In[10]))
				              +   0.026887*(REAL(In[-9])+REAL(In[9]))
				              +   0.0298618*(REAL(In[-8])+REAL(In[8]))
				              +   0.0327588*(REAL(In[-7])+REAL(In[7]))
				              +   0.035496*(REAL(In[-6])+REAL(In[6]))
				              +   0.0379899*(REAL(In[-5])+REAL(In[5]))
				              +   0.0401602*(REAL(In[-4])+REAL(In[4]))
				              +   0.0419335*(REAL(In[-3])+REAL(In[3]))
				              +   0.043248*(REAL(In[-2])+REAL(In[2]))
				              +   0.0440563*(REAL(In[-1])+REAL(In[1]))
				              +   0.0443291*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num94(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-36),-36,36,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num95 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num95(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.12732e-05*(REAL(In[-37])+REAL(In[37]))
				              +   1.75174e-05*(REAL(In[-36])+REAL(In[36]))
				              +   2.68935e-05*(REAL(In[-35])+REAL(In[35]))
				              +   4.07926e-05*(REAL(In[-34])+REAL(In[34]))
				              +   6.11324e-05*(REAL(In[-33])+REAL(In[33]))
				              +   9.05142e-05*(REAL(In[-32])+REAL(In[32]))
				              +   0.000132409*(REAL(In[-31])+REAL(In[31]))
				              +   0.00019137*(REAL(In[-30])+REAL(In[30]))
				              +   0.000273266*(REAL(In[-29])+REAL(In[29]))
				              +   0.000385525*(REAL(In[-28])+REAL(In[28]))
				              +   0.000537373*(REAL(In[-27])+REAL(In[27]))
				              +   0.000740039*(REAL(In[-26])+REAL(In[26]))
				              +   0.0010069*(REAL(In[-25])+REAL(In[25]))
				              +   0.00135356*(REAL(In[-24])+REAL(In[24]))
				              +   0.00179772*(REAL(In[-23])+REAL(In[23]))
				              +   0.00235898*(REAL(In[-22])+REAL(In[22]))
				              +   0.0030583*(REAL(In[-21])+REAL(In[21]))
				              +   0.00391734*(REAL(In[-20])+REAL(In[20]))
				              +   0.00495745*(REAL(In[-19])+REAL(In[19]))
				              +   0.00619842*(REAL(In[-18])+REAL(In[18]))
				              +   0.00765701*(REAL(In[-17])+REAL(In[17]))
				              +   0.00934529*(REAL(In[-16])+REAL(In[16]))
				              +   0.0112689*(REAL(In[-15])+REAL(In[15]))
				              +   0.0134254*(REAL(In[-14])+REAL(In[14]))
				              +   0.0158026*(REAL(In[-13])+REAL(In[13]))
				              +   0.0183774*(REAL(In[-12])+REAL(In[12]))
				              +   0.0211152*(REAL(In[-11])+REAL(In[11]))
				              +   0.0239696*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268834*(REAL(In[-9])+REAL(In[9]))
				              +   0.0297894*(REAL(In[-8])+REAL(In[8]))
				              +   0.0326134*(REAL(In[-7])+REAL(In[7]))
				              +   0.0352764*(REAL(In[-6])+REAL(In[6]))
				              +   0.0376989*(REAL(In[-5])+REAL(In[5]))
				              +   0.0398042*(REAL(In[-4])+REAL(In[4]))
				              +   0.0415226*(REAL(In[-3])+REAL(In[3]))
				              +   0.0427953*(REAL(In[-2])+REAL(In[2]))
				              +   0.0435775*(REAL(In[-1])+REAL(In[1]))
				              +   0.0438415*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num95(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-37),-37,37,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num96 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num96(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.33327e-05*(REAL(In[-37])+REAL(In[37]))
				              +   2.05212e-05*(REAL(In[-36])+REAL(In[36]))
				              +   3.12145e-05*(REAL(In[-35])+REAL(In[35]))
				              +   4.69222e-05*(REAL(In[-34])+REAL(In[34]))
				              +   6.97059e-05*(REAL(In[-33])+REAL(In[33]))
				              +   0.000102336*(REAL(In[-32])+REAL(In[32]))
				              +   0.000148477*(REAL(In[-31])+REAL(In[31]))
				              +   0.000212891*(REAL(In[-30])+REAL(In[30]))
				              +   0.000301664*(REAL(In[-29])+REAL(In[29]))
				              +   0.000422434*(REAL(In[-28])+REAL(In[28]))
				              +   0.000584607*(REAL(In[-27])+REAL(In[27]))
				              +   0.000799535*(REAL(In[-26])+REAL(In[26]))
				              +   0.00108064*(REAL(In[-25])+REAL(In[25]))
				              +   0.00144342*(REAL(In[-24])+REAL(In[24]))
				              +   0.00190534*(REAL(In[-23])+REAL(In[23]))
				              +   0.00248554*(REAL(In[-22])+REAL(In[22]))
				              +   0.00320434*(REAL(In[-21])+REAL(In[21]))
				              +   0.0040825*(REAL(In[-20])+REAL(In[20]))
				              +   0.00514023*(REAL(In[-19])+REAL(In[19]))
				              +   0.00639598*(REAL(In[-18])+REAL(In[18]))
				              +   0.00786504*(REAL(In[-17])+REAL(In[17]))
				              +   0.00955793*(REAL(In[-16])+REAL(In[16]))
				              +   0.0114788*(REAL(In[-15])+REAL(In[15]))
				              +   0.0136237*(REAL(In[-14])+REAL(In[14]))
				              +   0.0159796*(REAL(In[-13])+REAL(In[13]))
				              +   0.0185227*(REAL(In[-12])+REAL(In[12]))
				              +   0.0212184*(REAL(In[-11])+REAL(In[11]))
				              +   0.0240208*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268741*(REAL(In[-9])+REAL(In[9]))
				              +   0.0297131*(REAL(In[-8])+REAL(In[8]))
				              +   0.0324661*(REAL(In[-7])+REAL(In[7]))
				              +   0.0350576*(REAL(In[-6])+REAL(In[6]))
				              +   0.0374114*(REAL(In[-5])+REAL(In[5]))
				              +   0.0394542*(REAL(In[-4])+REAL(In[4]))
				              +   0.0411199*(REAL(In[-3])+REAL(In[3]))
				              +   0.0423526*(REAL(In[-2])+REAL(In[2]))
				              +   0.0431098*(REAL(In[-1])+REAL(In[1]))
				              +   0.0433653*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num96(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-37),-37,37,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num97 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num97(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.01635e-05*(REAL(In[-38])+REAL(In[38]))
				              +   1.56799e-05*(REAL(In[-37])+REAL(In[37]))
				              +   2.39124e-05*(REAL(In[-36])+REAL(In[36]))
				              +   3.60479e-05*(REAL(In[-35])+REAL(In[35]))
				              +   5.37175e-05*(REAL(In[-34])+REAL(In[34]))
				              +   7.9128e-05*(REAL(In[-33])+REAL(In[33]))
				              +   0.000115219*(REAL(In[-32])+REAL(In[32]))
				              +   0.000165842*(REAL(In[-31])+REAL(In[31]))
				              +   0.000235964*(REAL(In[-30])+REAL(In[30]))
				              +   0.000331875*(REAL(In[-29])+REAL(In[29]))
				              +   0.000461405*(REAL(In[-28])+REAL(In[28]))
				              +   0.000634116*(REAL(In[-27])+REAL(In[27]))
				              +   0.000861458*(REAL(In[-26])+REAL(In[26]))
				              +   0.00115685*(REAL(In[-25])+REAL(In[25]))
				              +   0.00153568*(REAL(In[-24])+REAL(In[24]))
				              +   0.00201512*(REAL(In[-23])+REAL(In[23]))
				              +   0.00261385*(REAL(In[-22])+REAL(In[22]))
				              +   0.0033515*(REAL(In[-21])+REAL(In[21]))
				              +   0.00424793*(REAL(In[-20])+REAL(In[20]))
				              +   0.00532222*(REAL(In[-19])+REAL(In[19]))
				              +   0.00659155*(REAL(In[-18])+REAL(In[18]))
				              +   0.00806976*(REAL(In[-17])+REAL(In[17]))
				              +   0.0097659*(REAL(In[-16])+REAL(In[16]))
				              +   0.0116827*(REAL(In[-15])+REAL(In[15]))
				              +   0.013815*(REAL(In[-14])+REAL(In[14]))
				              +   0.0161488*(REAL(In[-13])+REAL(In[13]))
				              +   0.0186598*(REAL(In[-12])+REAL(In[12]))
				              +   0.0213134*(REAL(In[-11])+REAL(In[11]))
				              +   0.0240644*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268583*(REAL(In[-9])+REAL(In[9]))
				              +   0.0296319*(REAL(In[-8])+REAL(In[8]))
				              +   0.0323161*(REAL(In[-7])+REAL(In[7]))
				              +   0.0348384*(REAL(In[-6])+REAL(In[6]))
				              +   0.0371258*(REAL(In[-5])+REAL(In[5]))
				              +   0.0391085*(REAL(In[-4])+REAL(In[4]))
				              +   0.0407236*(REAL(In[-3])+REAL(In[3]))
				              +   0.0419179*(REAL(In[-2])+REAL(In[2]))
				              +   0.0426512*(REAL(In[-1])+REAL(In[1]))
				              +   0.0428985*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num97(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-38),-38,38,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num98 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num98(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.19986e-05*(REAL(In[-38])+REAL(In[38]))
				              +   1.83419e-05*(REAL(In[-37])+REAL(In[37]))
				              +   2.77232e-05*(REAL(In[-36])+REAL(In[36]))
				              +   4.14312e-05*(REAL(In[-35])+REAL(In[35]))
				              +   6.12205e-05*(REAL(In[-34])+REAL(In[34]))
				              +   8.94441e-05*(REAL(In[-33])+REAL(In[33]))
				              +   0.000129208*(REAL(In[-32])+REAL(In[32]))
				              +   0.00018455*(REAL(In[-31])+REAL(In[31]))
				              +   0.00026063*(REAL(In[-30])+REAL(In[30]))
				              +   0.000363931*(REAL(In[-29])+REAL(In[29]))
				              +   0.000502456*(REAL(In[-28])+REAL(In[28]))
				              +   0.000685902*(REAL(In[-27])+REAL(In[27]))
				              +   0.000925787*(REAL(In[-26])+REAL(In[26]))
				              +   0.00123551*(REAL(In[-25])+REAL(In[25]))
				              +   0.00163029*(REAL(In[-24])+REAL(In[24]))
				              +   0.002127*(REAL(In[-23])+REAL(In[23]))
				              +   0.00274383*(REAL(In[-22])+REAL(In[22]))
				              +   0.00349969*(REAL(In[-21])+REAL(In[21]))
				              +   0.00441356*(REAL(In[-20])+REAL(In[20]))
				              +   0.00550342*(REAL(In[-19])+REAL(In[19]))
				              +   0.00678517*(REAL(In[-18])+REAL(In[18]))
				              +   0.00827131*(REAL(In[-17])+REAL(In[17]))
				              +   0.00996949*(REAL(In[-16])+REAL(In[16]))
				              +   0.0118811*(REAL(In[-15])+REAL(In[15]))
				              +   0.0139999*(REAL(In[-14])+REAL(In[14]))
				              +   0.0163109*(REAL(In[-13])+REAL(In[13]))
				              +   0.0187895*(REAL(In[-12])+REAL(In[12]))
				              +   0.0214013*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241017*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268374*(REAL(In[-9])+REAL(In[9]))
				              +   0.0295473*(REAL(In[-8])+REAL(In[8]))
				              +   0.0321648*(REAL(In[-7])+REAL(In[7]))
				              +   0.0346202*(REAL(In[-6])+REAL(In[6]))
				              +   0.0368436*(REAL(In[-5])+REAL(In[5]))
				              +   0.0387686*(REAL(In[-4])+REAL(In[4]))
				              +   0.040335*(REAL(In[-3])+REAL(In[3]))
				              +   0.0414926*(REAL(In[-2])+REAL(In[2]))
				              +   0.0422029*(REAL(In[-1])+REAL(In[1]))
				              +   0.0424424*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num98(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-38),-38,38,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num99 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num99(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.40881e-05*(REAL(In[-38])+REAL(In[38]))
				              +   2.13455e-05*(REAL(In[-37])+REAL(In[37]))
				              +   3.19853e-05*(REAL(In[-36])+REAL(In[36]))
				              +   4.74002e-05*(REAL(In[-35])+REAL(In[35]))
				              +   6.94702e-05*(REAL(In[-34])+REAL(In[34]))
				              +   0.000100694*(REAL(In[-33])+REAL(In[33]))
				              +   0.000144344*(REAL(In[-32])+REAL(In[32]))
				              +   0.000204636*(REAL(In[-31])+REAL(In[31]))
				              +   0.000286914*(REAL(In[-30])+REAL(In[30]))
				              +   0.00039784*(REAL(In[-29])+REAL(In[29]))
				              +   0.000545575*(REAL(In[-28])+REAL(In[28]))
				              +   0.000739925*(REAL(In[-27])+REAL(In[27]))
				              +   0.000992451*(REAL(In[-26])+REAL(In[26]))
				              +   0.00131649*(REAL(In[-25])+REAL(In[25]))
				              +   0.00172709*(REAL(In[-24])+REAL(In[24]))
				              +   0.00224079*(REAL(In[-23])+REAL(In[23]))
				              +   0.00287524*(REAL(In[-22])+REAL(In[22]))
				              +   0.00364867*(REAL(In[-21])+REAL(In[21]))
				              +   0.00457913*(REAL(In[-20])+REAL(In[20]))
				              +   0.00568355*(REAL(In[-19])+REAL(In[19]))
				              +   0.0069766*(REAL(In[-18])+REAL(In[18]))
				              +   0.00846946*(REAL(In[-17])+REAL(In[17]))
				              +   0.0101685*(REAL(In[-16])+REAL(In[16]))
				              +   0.0120738*(REAL(In[-15])+REAL(In[15]))
				              +   0.0141782*(REAL(In[-14])+REAL(In[14]))
				              +   0.0164658*(REAL(In[-13])+REAL(In[13]))
				              +   0.0189119*(REAL(In[-12])+REAL(In[12]))
				              +   0.021482*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241325*(REAL(In[-10])+REAL(In[10]))
				              +   0.0268113*(REAL(In[-9])+REAL(In[9]))
				              +   0.0294591*(REAL(In[-8])+REAL(In[8]))
				              +   0.0320119*(REAL(In[-7])+REAL(In[7]))
				              +   0.0344025*(REAL(In[-6])+REAL(In[6]))
				              +   0.0365643*(REAL(In[-5])+REAL(In[5]))
				              +   0.0384336*(REAL(In[-4])+REAL(In[4]))
				              +   0.0399534*(REAL(In[-3])+REAL(In[3]))
				              +   0.0410756*(REAL(In[-2])+REAL(In[2]))
				              +   0.041764*(REAL(In[-1])+REAL(In[1]))
				              +   0.041996*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num99(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-38),-38,38,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num100 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num100(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.08362e-05*(REAL(In[-39])+REAL(In[39]))
				              +   1.64553e-05*(REAL(In[-38])+REAL(In[38]))
				              +   2.47184e-05*(REAL(In[-37])+REAL(In[37]))
				              +   3.67301e-05*(REAL(In[-36])+REAL(In[36]))
				              +   5.39899e-05*(REAL(In[-35])+REAL(In[35]))
				              +   7.85037e-05*(REAL(In[-34])+REAL(In[34]))
				              +   0.000112916*(REAL(In[-33])+REAL(In[33]))
				              +   0.00016066*(REAL(In[-32])+REAL(In[32]))
				              +   0.000226125*(REAL(In[-31])+REAL(In[31]))
				              +   0.000314831*(REAL(In[-30])+REAL(In[30]))
				              +   0.000433603*(REAL(In[-29])+REAL(In[29]))
				              +   0.00059074*(REAL(In[-28])+REAL(In[28]))
				              +   0.000796135*(REAL(In[-27])+REAL(In[27]))
				              +   0.00106137*(REAL(In[-26])+REAL(In[26]))
				              +   0.00139969*(REAL(In[-25])+REAL(In[25]))
				              +   0.00182593*(REAL(In[-24])+REAL(In[24]))
				              +   0.00235628*(REAL(In[-23])+REAL(In[23]))
				              +   0.00300785*(REAL(In[-22])+REAL(In[22]))
				              +   0.00379815*(REAL(In[-21])+REAL(In[21]))
				              +   0.00474434*(REAL(In[-20])+REAL(In[20]))
				              +   0.0058623*(REAL(In[-19])+REAL(In[19]))
				              +   0.00716551*(REAL(In[-18])+REAL(In[18]))
				              +   0.00866392*(REAL(In[-17])+REAL(In[17]))
				              +   0.0103626*(REAL(In[-16])+REAL(In[16]))
				              +   0.0122606*(REAL(In[-15])+REAL(In[15]))
				              +   0.0143496*(REAL(In[-14])+REAL(In[14]))
				              +   0.0166134*(REAL(In[-13])+REAL(In[13]))
				              +   0.0190267*(REAL(In[-12])+REAL(In[12]))
				              +   0.0215554*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241566*(REAL(In[-10])+REAL(In[10]))
				              +   0.0267796*(REAL(In[-9])+REAL(In[9]))
				              +   0.029367*(REAL(In[-8])+REAL(In[8]))
				              +   0.0318568*(REAL(In[-7])+REAL(In[7]))
				              +   0.0341847*(REAL(In[-6])+REAL(In[6]))
				              +   0.036287*(REAL(In[-5])+REAL(In[5]))
				              +   0.0381028*(REAL(In[-4])+REAL(In[4]))
				              +   0.0395776*(REAL(In[-3])+REAL(In[3]))
				              +   0.0406659*(REAL(In[-2])+REAL(In[2]))
				              +   0.0413332*(REAL(In[-1])+REAL(In[1]))
				              +   0.0415581*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num100(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-39),-39,39,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num101 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num101(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.27026e-05*(REAL(In[-39])+REAL(In[39]))
				              +   1.91248e-05*(REAL(In[-38])+REAL(In[38]))
				              +   2.84897e-05*(REAL(In[-37])+REAL(In[37]))
				              +   4.19916e-05*(REAL(In[-36])+REAL(In[36]))
				              +   6.1238e-05*(REAL(In[-35])+REAL(In[35]))
				              +   8.83616e-05*(REAL(In[-34])+REAL(In[34]))
				              +   0.000126151*(REAL(In[-33])+REAL(In[33]))
				              +   0.000178198*(REAL(In[-32])+REAL(In[32]))
				              +   0.000249056*(REAL(In[-31])+REAL(In[31]))
				              +   0.000344412*(REAL(In[-30])+REAL(In[30]))
				              +   0.00047124*(REAL(In[-29])+REAL(In[29]))
				              +   0.000637955*(REAL(In[-28])+REAL(In[28]))
				              +   0.000854522*(REAL(In[-27])+REAL(In[27]))
				              +   0.0011325*(REAL(In[-26])+REAL(In[26]))
				              +   0.00148505*(REAL(In[-25])+REAL(In[25]))
				              +   0.00192676*(REAL(In[-24])+REAL(In[24]))
				              +   0.00247341*(REAL(In[-23])+REAL(In[23]))
				              +   0.0031416*(REAL(In[-22])+REAL(In[22]))
				              +   0.00394811*(REAL(In[-21])+REAL(In[21]))
				              +   0.00490921*(REAL(In[-20])+REAL(In[20]))
				              +   0.00603974*(REAL(In[-19])+REAL(In[19]))
				              +   0.00735207*(REAL(In[-18])+REAL(In[18]))
				              +   0.00885492*(REAL(In[-17])+REAL(In[17]))
				              +   0.0105522*(REAL(In[-16])+REAL(In[16]))
				              +   0.0124419*(REAL(In[-15])+REAL(In[15]))
				              +   0.014515*(REAL(In[-14])+REAL(In[14]))
				              +   0.0167544*(REAL(In[-13])+REAL(In[13]))
				              +   0.0191349*(REAL(In[-12])+REAL(In[12]))
				              +   0.0216225*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241753*(REAL(In[-10])+REAL(In[10]))
				              +   0.0267437*(REAL(In[-9])+REAL(In[9]))
				              +   0.0292722*(REAL(In[-8])+REAL(In[8]))
				              +   0.031701*(REAL(In[-7])+REAL(In[7]))
				              +   0.0339684*(REAL(In[-6])+REAL(In[6]))
				              +   0.0360132*(REAL(In[-5])+REAL(In[5]))
				              +   0.0377774*(REAL(In[-4])+REAL(In[4]))
				              +   0.0392092*(REAL(In[-3])+REAL(In[3]))
				              +   0.0402649*(REAL(In[-2])+REAL(In[2]))
				              +   0.040912*(REAL(In[-1])+REAL(In[1]))
				              +   0.04113*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num101(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-39),-39,39,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num102 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num102(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   9.81943e-06*(REAL(In[-40])+REAL(In[40]))
				              +   1.48151e-05*(REAL(In[-39])+REAL(In[39]))
				              +   2.21209e-05*(REAL(In[-38])+REAL(In[38]))
				              +   3.26873e-05*(REAL(In[-37])+REAL(In[37]))
				              +   4.78005e-05*(REAL(In[-36])+REAL(In[36]))
				              +   6.91774e-05*(REAL(In[-35])+REAL(In[35]))
				              +   9.90772e-05*(REAL(In[-34])+REAL(In[34]))
				              +   0.00014043*(REAL(In[-33])+REAL(In[33]))
				              +   0.000196982*(REAL(In[-32])+REAL(In[32]))
				              +   0.000273445*(REAL(In[-31])+REAL(In[31]))
				              +   0.000375657*(REAL(In[-30])+REAL(In[30]))
				              +   0.00051073*(REAL(In[-29])+REAL(In[29]))
				              +   0.000687177*(REAL(In[-28])+REAL(In[28]))
				              +   0.000915007*(REAL(In[-27])+REAL(In[27]))
				              +   0.00120575*(REAL(In[-26])+REAL(In[26]))
				              +   0.00157242*(REAL(In[-25])+REAL(In[25]))
				              +   0.00202936*(REAL(In[-24])+REAL(In[24]))
				              +   0.00259195*(REAL(In[-23])+REAL(In[23]))
				              +   0.00327621*(REAL(In[-22])+REAL(In[22]))
				              +   0.00409822*(REAL(In[-21])+REAL(In[21]))
				              +   0.00507336*(REAL(In[-20])+REAL(In[20]))
				              +   0.00621549*(REAL(In[-19])+REAL(In[19]))
				              +   0.00753586*(REAL(In[-18])+REAL(In[18]))
				              +   0.00904207*(REAL(In[-17])+REAL(In[17]))
				              +   0.010737*(REAL(In[-16])+REAL(In[16]))
				              +   0.0126175*(REAL(In[-15])+REAL(In[15]))
				              +   0.0146738*(REAL(In[-14])+REAL(In[14]))
				              +   0.0168884*(REAL(In[-13])+REAL(In[13]))
				              +   0.019236*(REAL(In[-12])+REAL(In[12]))
				              +   0.0216829*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241879*(REAL(In[-10])+REAL(In[10]))
				              +   0.0267029*(REAL(In[-9])+REAL(In[9]))
				              +   0.0291739*(REAL(In[-8])+REAL(In[8]))
				              +   0.0315435*(REAL(In[-7])+REAL(In[7]))
				              +   0.0337523*(REAL(In[-6])+REAL(In[6]))
				              +   0.0357416*(REAL(In[-5])+REAL(In[5]))
				              +   0.0374562*(REAL(In[-4])+REAL(In[4]))
				              +   0.0388464*(REAL(In[-3])+REAL(In[3]))
				              +   0.0398708*(REAL(In[-2])+REAL(In[2]))
				              +   0.0404985*(REAL(In[-1])+REAL(In[1]))
				              +   0.0407098*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num102(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-40),-40,40,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num103 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num103(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.1492e-05*(REAL(In[-40])+REAL(In[40]))
				              +   1.71958e-05*(REAL(In[-39])+REAL(In[39]))
				              +   2.54696e-05*(REAL(In[-38])+REAL(In[38]))
				              +   3.73413e-05*(REAL(In[-37])+REAL(In[37]))
				              +   5.41907e-05*(REAL(In[-36])+REAL(In[36]))
				              +   7.78448e-05*(REAL(In[-35])+REAL(In[35]))
				              +   0.000110689*(REAL(In[-34])+REAL(In[34]))
				              +   0.000155792*(REAL(In[-33])+REAL(In[33]))
				              +   0.000217049*(REAL(In[-32])+REAL(In[32]))
				              +   0.000299322*(REAL(In[-31])+REAL(In[31]))
				              +   0.00040859*(REAL(In[-30])+REAL(In[30]))
				              +   0.000552085*(REAL(In[-29])+REAL(In[29]))
				              +   0.000738402*(REAL(In[-28])+REAL(In[28]))
				              +   0.000977572*(REAL(In[-27])+REAL(In[27]))
				              +   0.00128107*(REAL(In[-26])+REAL(In[26]))
				              +   0.00166176*(REAL(In[-25])+REAL(In[25]))
				              +   0.00213368*(REAL(In[-24])+REAL(In[24]))
				              +   0.00271182*(REAL(In[-23])+REAL(In[23]))
				              +   0.00341163*(REAL(In[-22])+REAL(In[22]))
				              +   0.00424845*(REAL(In[-21])+REAL(In[21]))
				              +   0.00523683*(REAL(In[-20])+REAL(In[20]))
				              +   0.00638963*(REAL(In[-19])+REAL(In[19]))
				              +   0.00771705*(REAL(In[-18])+REAL(In[18]))
				              +   0.00922563*(REAL(In[-17])+REAL(In[17]))
				              +   0.0109172*(REAL(In[-16])+REAL(In[16]))
				              +   0.0127877*(REAL(In[-15])+REAL(In[15]))
				              +   0.0148267*(REAL(In[-14])+REAL(In[14]))
				              +   0.0170162*(REAL(In[-13])+REAL(In[13]))
				              +   0.0193309*(REAL(In[-12])+REAL(In[12]))
				              +   0.0217375*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241956*(REAL(In[-10])+REAL(In[10]))
				              +   0.0266583*(REAL(In[-9])+REAL(In[9]))
				              +   0.0290735*(REAL(In[-8])+REAL(In[8]))
				              +   0.0313856*(REAL(In[-7])+REAL(In[7]))
				              +   0.0335377*(REAL(In[-6])+REAL(In[6]))
				              +   0.0354735*(REAL(In[-5])+REAL(In[5]))
				              +   0.0371402*(REAL(In[-4])+REAL(In[4]))
				              +   0.0384905*(REAL(In[-3])+REAL(In[3]))
				              +   0.0394849*(REAL(In[-2])+REAL(In[2]))
				              +   0.0400939*(REAL(In[-1])+REAL(In[1]))
				              +   0.0402989*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num103(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-40),-40,40,0){}
};

#define HAS_REAL4_COMPILED_CONVOLUTIONS 

class cConvolSpec_REAL4_Num104 : public cConvolSpec<REAL4>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<REAL4> * duplicate() const { return new cConvolSpec_REAL4_Num104(*this); }
		void Convol(REAL4 *Out, const REAL4 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1.33837e-05*(REAL(In[-40])+REAL(In[40]))
				              +   1.98666e-05*(REAL(In[-39])+REAL(In[39]))
				              +   2.91962e-05*(REAL(In[-38])+REAL(In[38]))
				              +   4.24802e-05*(REAL(In[-37])+REAL(In[37]))
				              +   6.11933e-05*(REAL(In[-36])+REAL(In[36]))
				              +   8.72727e-05*(REAL(In[-35])+REAL(In[35]))
				              +   0.000123228*(REAL(In[-34])+REAL(In[34]))
				              +   0.000172266*(REAL(In[-33])+REAL(In[33]))
				              +   0.000238421*(REAL(In[-32])+REAL(In[32]))
				              +   0.000326699*(REAL(In[-31])+REAL(In[31]))
				              +   0.000443207*(REAL(In[-30])+REAL(In[30]))
				              +   0.000595284*(REAL(In[-29])+REAL(In[29]))
				              +   0.000791585*(REAL(In[-28])+REAL(In[28]))
				              +   0.00104215*(REAL(In[-27])+REAL(In[27]))
				              +   0.00135837*(REAL(In[-26])+REAL(In[26]))
				              +   0.00175292*(REAL(In[-25])+REAL(In[25]))
				              +   0.00223957*(REAL(In[-24])+REAL(In[24]))
				              +   0.00283285*(REAL(In[-23])+REAL(In[23]))
				              +   0.00354764*(REAL(In[-22])+REAL(In[22]))
				              +   0.00439858*(REAL(In[-21])+REAL(In[21]))
				              +   0.00539937*(REAL(In[-20])+REAL(In[20]))
				              +   0.00656192*(REAL(In[-19])+REAL(In[19]))
				              +   0.00789542*(REAL(In[-18])+REAL(In[18]))
				              +   0.00940539*(REAL(In[-17])+REAL(In[17]))
				              +   0.0110926*(REAL(In[-16])+REAL(In[16]))
				              +   0.0129524*(REAL(In[-15])+REAL(In[15]))
				              +   0.0149735*(REAL(In[-14])+REAL(In[14]))
				              +   0.0171377*(REAL(In[-13])+REAL(In[13]))
				              +   0.0194196*(REAL(In[-12])+REAL(In[12]))
				              +   0.0217863*(REAL(In[-11])+REAL(In[11]))
				              +   0.0241983*(REAL(In[-10])+REAL(In[10]))
				              +   0.0266099*(REAL(In[-9])+REAL(In[9]))
				              +   0.0289706*(REAL(In[-8])+REAL(In[8]))
				              +   0.031227*(REAL(In[-7])+REAL(In[7]))
				              +   0.0333242*(REAL(In[-6])+REAL(In[6]))
				              +   0.0352083*(REAL(In[-5])+REAL(In[5]))
				              +   0.0368289*(REAL(In[-4])+REAL(In[4]))
				              +   0.0381407*(REAL(In[-3])+REAL(In[3]))
				              +   0.0391063*(REAL(In[-2])+REAL(In[2]))
				              +   0.0396973*(REAL(In[-1])+REAL(In[1]))
				              +   0.0398963*(In[0])
                           );
				In++;
			}
		}

		cConvolSpec_REAL4_Num104(REAL8 * aFilter):cConvolSpec<REAL4>(aFilter-(-40),-40,40,0){}
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
	{
		REAL8 theCoeff[3] = {1.92875e-22,1,1.92875e-22};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num6(theCoeff) );
	}
	{
		REAL8 theCoeff[5] = {2.21649e-10,0.00383626,0.992327,0.00383626,2.21649e-10};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num7(theCoeff) );
	}
	{
		REAL8 theCoeff[5] = {3.42561e-06,0.0403876,0.919218,0.0403876,3.42561e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num8(theCoeff) );
	}
	{
		REAL8 theCoeff[5] = {0.000263865,0.106451,0.786571,0.106451,0.000263865};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num9(theCoeff) );
	}
	{
		REAL8 theCoeff[7] = {2.47381e-06,0.00256626,0.165524,0.663815,0.165524,0.00256626,2.47381e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num10(theCoeff) );
	}
	{
		REAL8 theCoeff[7] = {5.85246e-05,0.00961893,0.2054,0.569846,0.2054,0.00961893,5.85246e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num11(theCoeff) );
	}
	{
		REAL8 theCoeff[9] = {1.85839e-06,0.000440742,0.0219102,0.22831,0.498675,0.22831,0.0219102,0.000440742,1.85839e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num12(theCoeff) );
	}
	{
		REAL8 theCoeff[9] = {2.27688e-05,0.00171364,0.0375263,0.239103,0.443269,0.239103,0.0375263,0.00171364,2.27688e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num13(theCoeff) );
	}
	{
		REAL8 theCoeff[9] = {0.000133831,0.00443186,0.0539911,0.241971,0.398943,0.241971,0.0539911,0.00443186,0.000133831};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num14(theCoeff) );
	}
	{
		REAL8 theCoeff[11] = {1.18305e-05,0.000487696,0.00879777,0.0694505,0.239915,0.362675,0.239915,0.0694505,0.00879777,0.000487696,1.18305e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num15(theCoeff) );
	}
	{
		REAL8 theCoeff[11] = {5.64693e-05,0.00128524,0.014607,0.0828978,0.234927,0.332453,0.234927,0.0828978,0.014607,0.00128524,5.64693e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num16(theCoeff) );
	}
	{
		REAL8 theCoeff[13] = {7.26683e-06,0.000188248,0.00269858,0.0214073,0.0939743,0.228285,0.306879,0.228285,0.0939743,0.0214073,0.00269858,0.000188248,7.26683e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num17(theCoeff) );
	}
	{
		REAL8 theCoeff[13] = {2.92661e-05,0.000484226,0.00481008,0.0286865,0.102713,0.220797,0.284959,0.220797,0.102713,0.0286865,0.00481008,0.000484226,2.92661e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num18(theCoeff) );
	}
	{
		REAL8 theCoeff[15] = {4.96403e-06,8.92202e-05,0.00102819,0.00759733,0.035994,0.10934,0.212965,0.265962,0.212965,0.10934,0.035994,0.00759733,0.00102819,8.92202e-05,4.96403e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num19(theCoeff) );
	}
	{
		REAL8 theCoeff[15] = {1.73963e-05,0.000220373,0.00188891,0.0109552,0.0429915,0.114156,0.205101,0.249339,0.205101,0.114156,0.0429915,0.0109552,0.00188891,0.000220373,1.73963e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num20(theCoeff) );
	}
	{
		REAL8 theCoeff[15] = {4.88348e-05,0.000462931,0.00310476,0.0147321,0.049457,0.117467,0.197391,0.234674,0.197391,0.117467,0.049457,0.0147321,0.00310476,0.000462931,4.88348e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num21(theCoeff) );
	}
	{
		REAL8 theCoeff[17] = {1.13844e-05,0.000115245,0.000856823,0.00467864,0.0187632,0.0552652,0.119552,0.18994,0.221635,0.18994,0.119552,0.0552652,0.0187632,0.00467864,0.000856823,0.000115245,1.13844e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num22(theCoeff) );
	}
	{
		REAL8 theCoeff[17] = {2.96796e-05,0.000236991,0.0014345,0.00658217,0.0228946,0.0603663,0.120657,0.182813,0.209971,0.182813,0.120657,0.0603663,0.0228946,0.00658217,0.0014345,0.000236991,2.96796e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num23(theCoeff) );
	}
	{
		REAL8 theCoeff[19] = {7.99188e-06,6.69152e-05,0.000436342,0.00221593,0.00876416,0.0269955,0.0647589,0.120986,0.176033,0.199471,0.176033,0.120986,0.0647589,0.0269955,0.00876416,0.00221593,0.000436342,6.69152e-05,7.99188e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num24(theCoeff) );
	}
	{
		REAL8 theCoeff[19] = {1.95108e-05,0.000134076,0.000734422,0.00320673,0.0111609,0.030964,0.0684755,0.120707,0.169611,0.189973,0.169611,0.120707,0.0684755,0.030964,0.0111609,0.00320673,0.000734422,0.000134076,1.95108e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num25(theCoeff) );
	}
	{
		REAL8 theCoeff[19] = {4.21131e-05,0.000243851,0.00114842,0.00439894,0.0137046,0.0347257,0.071566,0.119959,0.163542,0.18134,0.163542,0.119959,0.071566,0.0347257,0.0137046,0.00439894,0.00114842,0.000243851,4.21131e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num26(theCoeff) );
	}
	{
		REAL8 theCoeff[21] = {1.36245e-05,8.20815e-05,0.000409328,0.00168967,0.00577342,0.0163293,0.0382302,0.0740878,0.118847,0.15781,0.173454,0.15781,0.118847,0.0740878,0.0382302,0.0163293,0.00577342,0.00168967,0.000409328,8.20815e-05,1.36245e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num27(theCoeff) );
	}
	{
		REAL8 theCoeff[21] = {2.82349e-05,0.000146916,0.000642623,0.00236289,0.00730354,0.0189768,0.0414492,0.0761046,0.117465,0.152407,0.166228,0.152407,0.117465,0.0761046,0.0414492,0.0189768,0.00730354,0.00236289,0.000642623,0.000146916,2.82349e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num28(theCoeff) );
	}
	{
		REAL8 theCoeff[23] = {9.97702e-06,5.35323e-05,0.000244762,0.000953639,0.00316619,0.00895784,0.0215965,0.0443685,0.0776747,0.115877,0.147309,0.159577,0.147309,0.115877,0.0776747,0.0443685,0.0215965,0.00895784,0.00316619,0.000953639,0.000244762,5.35323e-05,9.97702e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num29(theCoeff) );
	}
	{
		REAL8 theCoeff[23] = {1.99128e-05,9.41246e-05,0.000383732,0.0013493,0.00409208,0.0107037,0.024148,0.0469875,0.0788568,0.114143,0.142501,0.153441,0.142501,0.114143,0.0788568,0.0469875,0.024148,0.0107037,0.00409208,0.0013493,0.000383732,9.41246e-05,1.99128e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num30(theCoeff) );
	}
	{
		REAL8 theCoeff[23] = {3.67559e-05,0.000155187,0.000571225,0.0018331,0.00512851,0.012509,0.0265999,0.0493131,0.0797024,0.112307,0.137964,0.147759,0.137964,0.112307,0.0797024,0.0493131,0.0265999,0.012509,0.00512851,0.0018331,0.000571225,0.000155187,3.67559e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num31(theCoeff) );
	}
	{
		REAL8 theCoeff[25] = {1.46331e-05,6.34418e-05,0.000242114,0.000813335,0.00240505,0.00626015,0.0143433,0.0289282,0.0513567,0.0802563,0.110399,0.133677,0.14248,0.133677,0.110399,0.0802563,0.0513567,0.0289282,0.0143433,0.00626015,0.00240505,0.000813335,0.000242114,6.34418e-05,1.46331e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num32(theCoeff) );
	}
	{
		REAL8 theCoeff[25] = {2.63282e-05,0.000103344,0.000360169,0.00111452,0.00306218,0.0074702,0.0161806,0.0311183,0.0531369,0.0805633,0.108452,0.129628,0.137568,0.129628,0.108452,0.0805633,0.0531369,0.0311183,0.0161806,0.0074702,0.00306218,0.00111452,0.000360169,0.000103344,2.63282e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num33(theCoeff) );
	}
	{
		REAL8 theCoeff[27] = {1.11237e-05,4.46104e-05,0.000160091,0.000514096,0.00147729,0.00379869,0.00874068,0.0179971,0.0331593,0.0546704,0.0806574,0.106483,0.125795,0.132982,0.125795,0.106483,0.0806574,0.0546704,0.0331593,0.0179971,0.00874068,0.00379869,0.00147729,0.000514096,0.000160091,4.46104e-05,1.11237e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num34(theCoeff) );
	}
	{
		REAL8 theCoeff[27] = {1.9536e-05,7.17356e-05,0.000237379,0.000707876,0.00190231,0.00460693,0.0100543,0.0197742,0.0350473,0.0559781,0.0805731,0.104513,0.122168,0.128693,0.122168,0.104513,0.0805731,0.0559781,0.0350473,0.0197742,0.0100543,0.00460693,0.00190231,0.000707876,0.000237379,7.17356e-05,1.9536e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num35(theCoeff) );
	}
	{
		REAL8 theCoeff[27] = {3.25082e-05,0.000110189,0.000338743,0.000944477,0.00238837,0.00547772,0.0113943,0.0214962,0.0367812,0.0570791,0.0803374,0.102553,0.118731,0.124672,0.118731,0.102553,0.0803374,0.0570791,0.0367812,0.0214962,0.0113943,0.00547772,0.00238837,0.000944477,0.000338743,0.000110189,3.25082e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num36(theCoeff) );
	}
	{
		REAL8 theCoeff[29] = {1.49331e-05,5.15859e-05,0.000162567,0.000467362,0.00122573,0.00293262,0.00640084,0.0127449,0.0231504,0.0383618,0.0579909,0.0799724,0.10061,0.115468,0.120893,0.115468,0.10061,0.0799724,0.0579909,0.0383618,0.0231504,0.0127449,0.00640084,0.00293262,0.00122573,0.000467362,0.000162567,5.15859e-05,1.49331e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num37(theCoeff) );
	}
	{
		REAL8 theCoeff[29] = {2.44177e-05,7.85022e-05,0.000231468,0.000625938,0.0015524,0.00353107,0.00736614,0.0140931,0.0247288,0.0397952,0.058734,0.0795024,0.0986965,0.112371,0.117338,0.112371,0.0986965,0.0795024,0.058734,0.0397952,0.0247288,0.0140931,0.00736614,0.00353107,0.0015524,0.000625938,0.000231468,7.85022e-05,2.44177e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num38(theCoeff) );
	}
	{
		REAL8 theCoeff[31] = {1.17065e-05,3.82375e-05,0.000115107,0.000319343,0.000816513,0.00192405,0.00417845,0.008363,0.0154261,0.026224,0.0410855,0.0593233,0.0789422,0.0968146,0.109426,0.113985,0.109426,0.0968146,0.0789422,0.0593233,0.0410855,0.026224,0.0154261,0.008363,0.00417845,0.00192405,0.000816513,0.000319343,0.000115107,3.82375e-05,1.17065e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num39(theCoeff) );
	}
	{
		REAL8 theCoeff[31] = {1.88234e-05,5.76232e-05,0.000163301,0.000428418,0.00104049,0.00233935,0.00486905,0.00938172,0.0167344,0.027633,0.042241,0.0597766,0.0783101,0.0949716,0.106625,0.110819,0.106625,0.0949716,0.0783101,0.0597766,0.042241,0.027633,0.0167344,0.00938172,0.00486905,0.00233935,0.00104049,0.000428418,0.000163301,5.76232e-05,1.88234e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num40(theCoeff) );
	}
	{
		REAL8 theCoeff[31] = {2.90956e-05,8.39109e-05,0.00022495,0.000560569,0.00129852,0.00279605,0.00559653,0.0104128,0.0180092,0.0289532,0.0432689,0.060108,0.0776183,0.0931693,0.103958,0.107825,0.103958,0.0931693,0.0776183,0.060108,0.0432689,0.0289532,0.0180092,0.0104128,0.00559653,0.00279605,0.00129852,0.000560569,0.00022495,8.39109e-05,2.90956e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num41(theCoeff) );
	}
	{
		REAL8 theCoeff[33] = {1.48399e-05,4.34116e-05,0.000118496,0.000301806,0.000717257,0.00159055,0.00329111,0.00635422,0.0114474,0.0192431,0.0301834,0.0441758,0.060329,0.0768761,0.0914073,0.101413,0.104986,0.101413,0.0914073,0.0768761,0.060329,0.0441758,0.0301834,0.0192431,0.0114474,0.00635422,0.00329111,0.00159055,0.000717257,0.000301806,0.000118496,4.34116e-05,1.48399e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num42(theCoeff) );
	}
	{
		REAL8 theCoeff[33] = {2.26487e-05,6.27506e-05,0.000162794,0.000395465,0.000899546,0.00191595,0.00382115,0.00713591,0.0124782,0.0204315,0.0313254,0.0449718,0.0604546,0.0760967,0.089691,0.0989871,0.102295,0.0989871,0.089691,0.0760967,0.0604546,0.0449718,0.0313254,0.0204315,0.0124782,0.00713591,0.00382115,0.00191595,0.000899546,0.000395465,0.000162794,6.27506e-05,2.26487e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num43(theCoeff) );
	}
	{
		REAL8 theCoeff[35] = {1.19298e-05,3.34579e-05,8.81499e-05,0.000218173,0.000507268,0.00110797,0.00227342,0.00438213,0.007935,0.0134979,0.0215696,0.0323798,0.0456628,0.0604934,0.0752852,0.0880173,0.0966681,0.0997367,0.0966681,0.0880173,0.0752852,0.0604934,0.0456628,0.0323798,0.0215696,0.0134979,0.007935,0.00438213,0.00227342,0.00110797,0.000507268,0.000218173,8.81499e-05,3.34579e-05,1.19298e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num44(theCoeff) );
	}
	{
		REAL8 theCoeff[35] = {1.79849e-05,4.79946e-05,0.000120682,0.000285928,0.000638316,0.0013427,0.00266126,0.00497004,0.00874575,0.014501,0.022655,0.03335,0.0462584,0.0604575,0.0744517,0.08639,0.0944532,0.0973048,0.0944532,0.08639,0.0744517,0.0604575,0.0462584,0.03335,0.022655,0.014501,0.00874575,0.00497004,0.00266126,0.0013427,0.000638316,0.000285928,0.000120682,4.79946e-05,1.79849e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num45(theCoeff) );
	}
	{
		REAL8 theCoeff[35] = {2.63089e-05,6.70395e-05,0.000161413,0.00036722,0.000789396,0.0016034,0.00307731,0.00558059,0.00956245,0.0154824,0.0236857,0.0342386,0.0467655,0.0603552,0.0736011,0.0848074,0.0923344,0.0949891,0.0923344,0.0848074,0.0736011,0.0603552,0.0467655,0.0342386,0.0236857,0.0154824,0.00956245,0.00558059,0.00307731,0.0016034,0.000789396,0.00036722,0.000161413,6.70395e-05,2.63089e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num46(theCoeff) );
	}
	{
		REAL8 theCoeff[37] = {1.45339e-05,3.74474e-05,9.14061e-05,0.000211369,0.000463039,0.000960962,0.00188933,0.00351901,0.00620933,0.0103796,0.0164373,0.0246599,0.035048,0.0471898,0.0601927,0.0727365,0.0832669,0.0903035,0.0927788,0.0903035,0.0832669,0.0727365,0.0601927,0.0471898,0.035048,0.0246599,0.0164373,0.0103796,0.00620933,0.00351901,0.00188933,0.000960962,0.000463039,0.000211369,9.14061e-05,3.74474e-05,1.45339e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num47(theCoeff) );
	}
	{
		REAL8 theCoeff[37] = {2.10568e-05,5.19951e-05,0.000121927,0.000271522,0.000574218,0.00115323,0.0021995,0.0039838,0.00685236,0.0111931,0.0173631,0.0255782,0.0357834,0.0475402,0.0599802,0.0718657,0.0817717,0.0883593,0.090671,0.0883593,0.0817717,0.0718657,0.0599802,0.0475402,0.0357834,0.0255782,0.0173631,0.0111931,0.00685236,0.0039838,0.0021995,0.00115323,0.000574218,0.000271522,0.000121927,5.19951e-05,2.10568e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num48(theCoeff) );
	}
	{
		REAL8 theCoeff[37] = {2.97412e-05,7.05788e-05,0.00015942,0.000342742,0.000701364,0.00136607,0.00253254,0.00446881,0.00750554,0.0119984,0.0182567,0.0264406,0.0364481,0.0478224,0.0597229,0.0709911,0.0803195,0.0864949,0.0886572,0.0864949,0.0803195,0.0709911,0.0597229,0.0478224,0.0364481,0.0264406,0.0182567,0.0119984,0.00750554,0.00446881,0.00253254,0.00136607,0.000701364,0.000342742,0.00015942,7.05788e-05,2.97412e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num49(theCoeff) );
	}
	{
		REAL8 theCoeff[39] = {1.71208e-05,4.10415e-05,9.38422e-05,0.000204668,0.000425771,0.000844849,0.00159903,0.00288676,0.00497096,0.00816481,0.0127917,0.0191154,0.0272468,0.0370445,0.0480405,0.0594247,0.0701136,0.0789065,0.0847031,0.0867285,0.0847031,0.0789065,0.0701136,0.0594247,0.0480405,0.0370445,0.0272468,0.0191154,0.0127917,0.00816481,0.00497096,0.00288676,0.00159903,0.000844849,0.000425771,0.000204668,9.38422e-05,4.10415e-05,1.71208e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num50(theCoeff) );
	}
	{
		REAL8 theCoeff[39] = {2.39969e-05,5.54458e-05,0.00012244,0.000258414,0.000521255,0.0010049,0.00185156,0.00326054,0.0054876,0.00882703,0.0135702,0.0199388,0.0279995,0.0375787,0.0482029,0.0590941,0.0692397,0.0775364,0.0829843,0.0848841,0.0829843,0.0775364,0.0692397,0.0590941,0.0482029,0.0375787,0.0279995,0.0199388,0.0135702,0.00882703,0.0054876,0.00326054,0.00185156,0.0010049,0.000521255,0.000258414,0.00012244,5.54458e-05,2.39969e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num51(theCoeff) );
	}
	{
		REAL8 theCoeff[41] = {1.41176e-05,3.29099e-05,7.34588e-05,0.000157004,0.000321314,0.000629649,0.00118146,0.00212269,0.0036518,0.00601557,0.0094885,0.0143307,0.0207248,0.0286987,0.0380526,0.0483122,0.0587327,0.0683682,0.076204,0.0813303,0.0831145,0.0813303,0.076204,0.0683682,0.0587327,0.0483122,0.0380526,0.0286987,0.0207248,0.0143307,0.0094885,0.00601557,0.0036518,0.00212269,0.00118146,0.000629649,0.000321314,0.000157004,7.34588e-05,3.29099e-05,1.41176e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num52(theCoeff) );
	}
	{
		REAL8 theCoeff[41] = {1.96387e-05,4.42415e-05,9.56003e-05,0.000198153,0.000393961,0.000751309,0.00137434,0.00241148,0.00405868,0.00655236,0.0101467,0.0150716,0.0214738,0.0293473,0.0384717,0.0483755,0.0583474,0.067504,0.0749117,0.0797411,0.0814191,0.0797411,0.0749117,0.067504,0.0583474,0.0483755,0.0384717,0.0293473,0.0214738,0.0150716,0.0101467,0.00655236,0.00405868,0.00241148,0.00137434,0.000751309,0.000393961,0.000198153,9.56003e-05,4.42415e-05,1.96387e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num53(theCoeff) );
	}
	{
		REAL8 theCoeff[41] = {2.67671e-05,5.83917e-05,0.000122385,0.000246454,0.000476837,0.000886405,0.00158315,0.0027167,0.00447909,0.0070952,0.0107986,0.0157907,0.0221851,0.0299467,0.0388388,0.0483961,0.0579406,0.0666476,0.073657,0.0782117,0.0797917,0.0782117,0.073657,0.0666476,0.0579406,0.0483961,0.0388388,0.0299467,0.0221851,0.0157907,0.0107986,0.0070952,0.00447909,0.0027167,0.00158315,0.000886405,0.000476837,0.000246454,0.000122385,5.83917e-05,2.67671e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num54(theCoeff) );
	}
	{
		REAL8 theCoeff[43] = {1.62785e-05,3.58021e-05,7.57711e-05,0.000154313,0.000302415,0.000570305,0.00103494,0.00180727,0.00303694,0.00491079,0.00764133,0.0114417,0.0164859,0.0228581,0.0304978,0.0391562,0.0483766,0.0575139,0.065798,0.0724363,0.0767365,0.0782259,0.0767365,0.0724363,0.065798,0.0575139,0.0483766,0.0391562,0.0304978,0.0228581,0.0164859,0.0114417,0.00764133,0.00491079,0.00303694,0.00180727,0.00103494,0.000570305,0.000302415,0.000154313,7.57711e-05,3.58021e-05,1.62785e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num55(theCoeff) );
	}
	{
		REAL8 theCoeff[43] = {2.20511e-05,4.70635e-05,9.68005e-05,0.000191871,0.000366505,0.000674668,0.00119685,0.00204609,0.00337094,0.005352,0.00818879,0.0120743,0.0171571,0.0234944,0.0310044,0.0394294,0.0483233,0.0570732,0.0649601,0.0712524,0.0753167,0.0767223,0.0753167,0.0712524,0.0649601,0.0570732,0.0483233,0.0394294,0.0310044,0.0234944,0.0171571,0.0120743,0.00818879,0.005352,0.00337094,0.00204609,0.00119685,0.000674668,0.000366505,0.000191871,9.68005e-05,4.70635e-05,2.20511e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num56(theCoeff) );
	}
	{
		REAL8 theCoeff[45] = {1.36497e-05,2.93449e-05,6.08806e-05,0.000121889,0.0002355,0.000439091,0.000790057,0.00137183,0.0022987,0.0037171,0.00580048,0.00873502,0.0126941,0.0178025,0.0240933,0.0314669,0.0396596,0.0482373,0.0566183,0.0641313,0.0701006,0.0739457,0.0752737,0.0739457,0.0701006,0.0641313,0.0566183,0.0482373,0.0396596,0.0314669,0.0240933,0.0178025,0.0126941,0.00873502,0.00580048,0.0037171,0.0022987,0.00137183,0.000790057,0.000439091,0.0002355,0.000121889,6.08806e-05,2.93449e-05,1.36497e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num57(theCoeff) );
	}
	{
		REAL8 theCoeff[45] = {1.83782e-05,3.8416e-05,7.75942e-05,0.000151444,0.000285616,0.000520497,0.000916561,0.00155959,0.00256429,0.00407407,0.00625457,0.00927839,0.0133001,0.0184222,0.0246568,0.0318888,0.0398517,0.0481239,0.0561541,0.0633153,0.068983,0.0726244,0.0738804,0.0726244,0.068983,0.0633153,0.0561541,0.0481239,0.0398517,0.0318888,0.0246568,0.0184222,0.0133001,0.00927839,0.00625457,0.00407407,0.00256429,0.00155959,0.000916561,0.000520497,0.000285616,0.000151444,7.75942e-05,3.8416e-05,1.83782e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num58(theCoeff) );
	}
	{
		REAL8 theCoeff[45] = {2.43338e-05,4.95315e-05,9.75432e-05,0.000185847,0.000342576,0.000610945,0.00105412,0.00175963,0.00284181,0.0044403,0.00671232,0.00981695,0.0138907,0.0190158,0.0251853,0.0322718,0.0400076,0.047985,0.0556815,0.0625115,0.0678972,0.0713489,0.072538,0.0713489,0.0678972,0.0625115,0.0556815,0.047985,0.0400076,0.0322718,0.0251853,0.0190158,0.0138907,0.00981695,0.00671232,0.0044403,0.00284181,0.00175963,0.00105412,0.000610945,0.000342576,0.000185847,9.75432e-05,4.95315e-05,2.43338e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num59(theCoeff) );
	}
	{
		REAL8 theCoeff[47] = {1.54796e-05,3.17215e-05,6.29652e-05,0.000121059,0.000225449,0.000406675,0.000710558,0.00120255,0.00197132,0.00313014,0.00481416,0.00717181,0.0103488,0.0144644,0.0195822,0.0256789,0.0326168,0.0401289,0.0478217,0.0552006,0.0617184,0.06684,0.0701147,0.0712416,0.0701147,0.06684,0.0617184,0.0552006,0.0478217,0.0401289,0.0326168,0.0256789,0.0195822,0.0144644,0.0103488,0.00717181,0.00481416,0.00313014,0.00197132,0.00120255,0.000710558,0.000406675,0.000225449,0.000121059,6.29652e-05,3.17215e-05,1.54796e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num60(theCoeff) );
	}
	{
		REAL8 theCoeff[47] = {2.03927e-05,4.07598e-05,7.89993e-05,0.000148473,0.000270585,0.000478183,0.000819439,0.00136167,0.00219412,0.00342833,0.00519443,0.00763178,0.0108729,0.015021,0.0201227,0.0261401,0.0329275,0.0402202,0.0476391,0.0547161,0.0609396,0.0658138,0.0689235,0.0699924,0.0689235,0.0658138,0.0609396,0.0547161,0.0476391,0.0402202,0.0329275,0.0261401,0.0201227,0.015021,0.0108729,0.00763178,0.00519443,0.00342833,0.00219412,0.00136167,0.000819439,0.000478183,0.000270585,0.000148473,7.89993e-05,4.07598e-05,2.03927e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num61(theCoeff) );
	}
	{
		REAL8 theCoeff[49] = {1.31642e-05,2.64716e-05,5.16722e-05,9.79092e-05,0.000180086,0.000321533,0.000557266,0.000937537,0.0015311,0.00242723,0.00373513,0.00557946,0.00809036,0.0113876,0.0155593,0.0206364,0.0265687,0.0332044,0.040282,0.0474369,0.0542265,0.0601723,0.0648145,0.06777,0.0687848,0.06777,0.0648145,0.0601723,0.0542265,0.0474369,0.040282,0.0332044,0.0265687,0.0206364,0.0155593,0.0113876,0.00809036,0.00557946,0.00373513,0.00242723,0.0015311,0.000937537,0.000557266,0.000321533,0.000180086,9.79092e-05,5.16722e-05,2.64716e-05,1.31642e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num62(theCoeff) );
	}
	{
		REAL8 theCoeff[49] = {1.72561e-05,3.38948e-05,6.46913e-05,0.000119973,0.000216194,0.000378555,0.000644076,0.0010648,0.00171051,0.00266997,0.00404958,0.00596813,0.00854653,0.0118923,0.0160792,0.0211246,0.0269672,0.0334508,0.0403182,0.0472194,0.0537357,0.0594196,0.0638439,0.0666552,0.0676195,0.0666552,0.0638439,0.0594196,0.0537357,0.0472194,0.0403182,0.0334508,0.0269672,0.0211246,0.0160792,0.0118923,0.00854653,0.00596813,0.00404958,0.00266997,0.00171051,0.0010648,0.000644076,0.000378555,0.000216194,0.000119973,6.46913e-05,3.38948e-05,1.72561e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num63(theCoeff) );
	}
	{
		REAL8 theCoeff[49] = {2.2306e-05,4.28469e-05,8.00486e-05,0.000145453,0.000257058,0.000441849,0.000738674,0.00120107,0.00189941,0.00292151,0.0043705,0.00635905,0.00899889,0.0123857,0.0165802,0.0215872,0.0273362,0.0336679,0.0403302,0.0469874,0.0532437,0.0586801,0.0628999,0.0655761,0.0664933,0.0655761,0.0628999,0.0586801,0.0532437,0.0469874,0.0403302,0.0336679,0.0273362,0.0215872,0.0165802,0.0123857,0.00899889,0.00635905,0.0043705,0.00292151,0.00189941,0.00120107,0.000738674,0.000441849,0.000257058,0.000145453,8.00486e-05,4.28469e-05,2.2306e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num64(theCoeff) );
	}
	{
		REAL8 theCoeff[51] = {1.47321e-05,2.84588e-05,5.35174e-05,9.79722e-05,0.000174598,0.000302903,0.00051156,0.000841044,0.00134607,0.00209724,0.00318094,0.00469668,0.0067508,0.00944601,0.0128668,0.0170616,0.0220241,0.0276762,0.0338565,0.0403187,0.0467412,0.0527498,0.0579524,0.0619797,0.0645293,0.0654022,0.0645293,0.0619797,0.0579524,0.0527498,0.0467412,0.0403187,0.0338565,0.0276762,0.0220241,0.0170616,0.0128668,0.00944601,0.0067508,0.00469668,0.00318094,0.00209724,0.00134607,0.000841044,0.00051156,0.000302903,0.000174598,9.79722e-05,5.35174e-05,2.84588e-05,1.47321e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num65(theCoeff) );
	}
	{
		REAL8 theCoeff[51] = {1.89634e-05,3.58687e-05,6.61026e-05,0.000118692,0.000207649,0.000353947,0.000587826,0.000951178,0.0014996,0.00230353,0.00344756,0.00502728,0.00714259,0.00988736,0.0133354,0.0175241,0.0224371,0.0279898,0.03402,0.0402876,0.0464847,0.0522578,0.0572393,0.0610856,0.0635164,0.064348,0.0635164,0.0610856,0.0572393,0.0522578,0.0464847,0.0402876,0.03402,0.0279898,0.0224371,0.0175241,0.0133354,0.00988736,0.00714259,0.00502728,0.00344756,0.00230353,0.0014996,0.000951178,0.000587826,0.000353947,0.000207649,0.000118692,6.61026e-05,3.58687e-05,1.89634e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num66(theCoeff) );
	}
	{
		REAL8 theCoeff[53] = {1.26803e-05,2.41076e-05,4.46928e-05,8.07938e-05,0.000142422,0.000244812,0.000410344,0.000670687,0.00106893,0.00166126,0.00251758,0.00372038,0.00536103,0.007533,0.0103215,0.0137905,0.0179669,0.0228256,0.0282769,0.0341584,0.0402366,0.0462172,0.0517658,0.0565381,0.0602138,0.062533,0.0633258,0.062533,0.0602138,0.0565381,0.0517658,0.0462172,0.0402366,0.0341584,0.0282769,0.0228256,0.0179669,0.0137905,0.0103215,0.007533,0.00536103,0.00372038,0.00251758,0.00166126,0.00106893,0.000670687,0.000410344,0.000244812,0.000142422,8.07938e-05,4.46928e-05,2.41076e-05,1.26803e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num67(theCoeff) );
	}
	{
		REAL8 theCoeff[53] = {1.62543e-05,3.02929e-05,5.50949e-05,9.77867e-05,0.000169373,0.000286291,0.000472244,0.000760191,0.0011942,0.00183074,0.00273889,0.00399871,0.0056972,0.00792137,0.0107482,0.0142321,0.0183908,0.0231914,0.0285399,0.0342747,0.0401692,0.0459419,0.0512769,0.0558511,0.0593662,0.0615805,0.0623369,0.0615805,0.0593662,0.0558511,0.0512769,0.0459419,0.0401692,0.0342747,0.0285399,0.0231914,0.0183908,0.0142321,0.0107482,0.00792137,0.0056972,0.00399871,0.00273889,0.00183074,0.0011942,0.000760191,0.000472244,0.000286291,0.000169373,9.77867e-05,5.50949e-05,3.02929e-05,1.62543e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num68(theCoeff) );
	}
	{
		REAL8 theCoeff[53] = {2.05902e-05,3.76512e-05,6.72385e-05,0.000117268,0.000199738,0.000332247,0.00053974,0.000856304,0.00132676,0.0020076,0.00296677,0.00428165,0.00603474,0.00830668,0.0111665,0.0146598,0.0187957,0.0235348,0.0287795,0.0343698,0.0400859,0.045659,0.0507905,0.0551772,0.0585407,0.0606564,0.0613785,0.0606564,0.0585407,0.0551772,0.0507905,0.045659,0.0400859,0.0343698,0.0287795,0.0235348,0.0187957,0.0146598,0.0111665,0.00830668,0.00603474,0.00428165,0.00296677,0.0020076,0.00132676,0.000856304,0.00053974,0.000332247,0.000199738,0.000117268,6.72385e-05,3.76512e-05,2.05902e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num69(theCoeff) );
	}
	{
		REAL8 theCoeff[55] = {1.40379e-05,2.57934e-05,4.63175e-05,8.12851e-05,0.000139414,0.000233686,0.000382814,0.000612877,0.000958935,0.00146634,0.00219134,0.00320048,0.00456826,0.0063726,0.00868784,0.0115754,0.0150728,0.0191813,0.0238558,0.028996,0.034444,0.039987,0.0453685,0.0503059,0.0545148,0.057735,0.0597578,0.0604476,0.0597578,0.057735,0.0545148,0.0503059,0.0453685,0.039987,0.034444,0.028996,0.0238558,0.0191813,0.0150728,0.0115754,0.00868784,0.0063726,0.00456826,0.00320048,0.00219134,0.00146634,0.000958935,0.000612877,0.000382814,0.000233686,0.000139414,8.12851e-05,4.63175e-05,2.57934e-05,1.40379e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num70(theCoeff) );
	}
	{
		REAL8 theCoeff[55] = {1.77193e-05,3.19761e-05,5.64324e-05,9.73995e-05,0.000164403,0.000271388,0.000438122,0.000691712,0.00106802,0.00161273,0.00238159,0.00343953,0.00485798,0.00671024,0.00906454,0.0119751,0.0154717,0.0195488,0.0241563,0.029192,0.0345004,0.0398758,0.0450733,0.0498259,0.0538662,0.0569513,0.0588864,0.059546,0.0588864,0.0569513,0.0538662,0.0498259,0.0450733,0.0398758,0.0345004,0.029192,0.0241563,0.0195488,0.0154717,0.0119751,0.00906454,0.00671024,0.00485798,0.00343953,0.00238159,0.00161273,0.00106802,0.000691712,0.000438122,0.000271388,0.000164403,9.73995e-05,5.64324e-05,3.19761e-05,1.77193e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num71(theCoeff) );
	}
	{
		REAL8 theCoeff[57] = {1.22089e-05,2.21291e-05,3.92514e-05,6.81328e-05,0.000115735,0.000192389,0.000312972,0.000498238,0.000776205,0.00118338,0.00176555,0.00257776,0.0036831,0.00514983,0.00704659,0.00943569,0.0123645,0.0158557,0.0198977,0.024436,0.0293672,0.0345386,0.0397515,0.0447724,0.0493487,0.053229,0.0561861,0.0580386,0.0586696,0.0580386,0.0561861,0.053229,0.0493487,0.0447724,0.0397515,0.0345386,0.0293672,0.024436,0.0198977,0.0158557,0.0123645,0.00943569,0.00704659,0.00514983,0.0036831,0.00257776,0.00176555,0.00118338,0.000776205,0.000498238,0.000312972,0.000192389,0.000115735,6.81328e-05,3.92514e-05,2.21291e-05,1.22089e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num72(theCoeff) );
	}
	{
		REAL8 theCoeff[57] = {1.53562e-05,2.73614e-05,4.77385e-05,8.156e-05,0.000136447,0.000223527,0.000358569,0.00056324,0.000866349,0.00130488,0.00192454,0.00277945,0.00393071,0.00544328,0.00738123,0.00980109,0.0127438,0.0162256,0.0202293,0.0246967,0.029524,0.0345613,0.039617,0.0444685,0.0488766,0.0526051,0.0554412,0.0572157,0.0578198,0.0572157,0.0554412,0.0526051,0.0488766,0.0444685,0.039617,0.0345613,0.029524,0.0246967,0.0202293,0.0162256,0.0127438,0.00980109,0.00738123,0.00544328,0.00393071,0.00277945,0.00192454,0.00130488,0.000866349,0.00056324,0.000358569,0.000223527,0.000136447,8.156e-05,4.77385e-05,2.73614e-05,1.53562e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num73(theCoeff) );
	}
	{
		REAL8 theCoeff[57] = {1.91195e-05,3.3513e-05,5.75554e-05,9.68494e-05,0.000159678,0.000257946,0.000408272,0.00063315,0.000962059,0.0014323,0.0020893,0.00298611,0.00418166,0.00573756,0.00771335,0.01016,0.0131125,0.016581,0.0205435,0.0249387,0.0296627,0.0345688,0.0394726,0.0441614,0.0484091,0.0519934,0.0547149,0.0564158,0.0569944,0.0564158,0.0547149,0.0519934,0.0484091,0.0441614,0.0394726,0.0345688,0.0296627,0.0249387,0.0205435,0.016581,0.0131125,0.01016,0.00771335,0.00573756,0.00418166,0.00298611,0.0020893,0.0014323,0.000962059,0.00063315,0.000408272,0.000257946,0.000159678,9.68494e-05,5.75554e-05,3.3513e-05,1.91195e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num74(theCoeff) );
	}
	{
		REAL8 theCoeff[59] = {1.33954e-05,2.35771e-05,4.06826e-05,6.88197e-05,0.00011413,0.000185556,0.000295756,0.000462143,0.000707953,0.0010632,0.00156536,0.00225942,0.00319715,0.00443521,0.00603185,0.00804214,0.0105118,0.01347,0.0169217,0.0208403,0.0251622,0.0297836,0.0345615,0.039318,0.0438506,0.047945,0.0513922,0.0540051,0.0556363,0.0561909,0.0556363,0.0540051,0.0513922,0.047945,0.0438506,0.039318,0.0345615,0.0297836,0.0251622,0.0208403,0.0169217,0.01347,0.0105118,0.00804214,0.00603185,0.00443521,0.00319715,0.00225942,0.00156536,0.0010632,0.000707953,0.000462143,0.000295756,0.000185556,0.00011413,6.88197e-05,4.06826e-05,2.35771e-05,1.33954e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num75(theCoeff) );
	}
	{
		REAL8 theCoeff[59] = {1.66271e-05,2.88123e-05,4.89736e-05,8.16523e-05,0.000133536,0.000214214,0.000337071,0.000520257,0.000787655,0.00116971,0.00170388,0.00243459,0.00341219,0.00469098,0.00632581,0.00836741,0.0108565,0.0138168,0.0172485,0.0211211,0.025369,0.0298891,0.0345418,0.039156,0.0435387,0.047487,0.0508038,0.0533139,0.0548791,0.0554109,0.0548791,0.0533139,0.0508038,0.047487,0.0435387,0.039156,0.0345418,0.0298891,0.025369,0.0211211,0.0172485,0.0138168,0.0108565,0.00836741,0.00632581,0.00469098,0.00341219,0.00243459,0.00170388,0.00116971,0.000787655,0.000520257,0.000337071,0.000214214,0.000133536,8.16523e-05,4.89736e-05,2.88123e-05,1.66271e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num76(theCoeff) );
	}
	{
		REAL8 theCoeff[61] = {1.1756e-05,2.04491e-05,3.49092e-05,5.84864e-05,9.61657e-05,0.00015518,0.000245755,0.000381961,0.000582621,0.000872175,0.00128136,0.00184752,0.00261431,0.00363058,0.00494818,0.00661858,0.0086883,0.0111932,0.0141522,0.0175609,0.0213854,0.0255587,0.0299786,0.0345091,0.0389858,0.0432244,0.047033,0.0502257,0.0526381,0.0541408,0.0546512,0.0541408,0.0526381,0.0502257,0.047033,0.0432244,0.0389858,0.0345091,0.0299786,0.0255587,0.0213854,0.0175609,0.0141522,0.0111932,0.0086883,0.00661858,0.00494818,0.00363058,0.00261431,0.00184752,0.00128136,0.000872175,0.000582621,0.000381961,0.000245755,0.00015518,9.61657e-05,5.84864e-05,3.49092e-05,2.04491e-05,1.1756e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num77(theCoeff) );
	}
	{
		REAL8 theCoeff[61] = {1.4548e-05,2.49324e-05,4.19559e-05,6.93254e-05,0.000112476,0.000179183,0.000280287,0.000430506,0.000649268,0.000961474,0.00139804,0.00199605,0.00279829,0.00385198,0.00520647,0.0069099,0.0090047,0.0115222,0.0144768,0.0178598,0.0216347,0.0257332,0.0300543,0.0344658,0.0388096,0.04291,0.0465852,0.0496599,0.0519796,0.0534231,0.0539131,0.0534231,0.0519796,0.0496599,0.0465852,0.04291,0.0388096,0.0344658,0.0300543,0.0257332,0.0216347,0.0178598,0.0144768,0.0115222,0.0090047,0.0069099,0.00520647,0.00385198,0.00279829,0.00199605,0.00139804,0.000961474,0.000649268,0.000430506,0.000280287,0.000179183,0.000112476,6.93254e-05,4.19559e-05,2.49324e-05,1.4548e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num78(theCoeff) );
	}
	{
		REAL8 theCoeff[61] = {1.78449e-05,3.01493e-05,5.00402e-05,8.15908e-05,0.00013069,0.000205647,0.000317893,0.000482748,0.000720175,0.00105544,0.00151954,0.00214915,0.00298608,0.00407582,0.00546522,0.00719913,0.00931605,0.011843,0.0147901,0.0181452,0.021869,0.0258927,0.0301165,0.034412,0.0386274,0.0425951,0.0461427,0.049105,0.0513367,0.0527241,0.0531948,0.0527241,0.0513367,0.049105,0.0461427,0.0425951,0.0386274,0.034412,0.0301165,0.0258927,0.021869,0.0181452,0.0147901,0.011843,0.00931605,0.00719913,0.00546522,0.00407582,0.00298608,0.00214915,0.00151954,0.00105544,0.000720175,0.000482748,0.000317893,0.000205647,0.00013069,8.15908e-05,5.00402e-05,3.01493e-05,1.78449e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num79(theCoeff) );
	}
	{
		REAL8 theCoeff[63] = {1.28014e-05,2.17062e-05,3.61737e-05,5.92493e-05,9.53793e-05,0.000150906,0.00023466,0.000358636,0.000538703,0.00079529,0.00115394,0.00164559,0.00230643,0.00317718,0.00430153,0.00572382,0.00748565,0.00962175,0.0121551,0.015092,0.0184168,0.0220884,0.0260372,0.0301651,0.0343476,0.0384388,0.042279,0.0457046,0.0485597,0.0507076,0.0520417,0.0524942,0.0520417,0.0507076,0.0485597,0.0457046,0.042279,0.0384388,0.0343476,0.0301651,0.0260372,0.0220884,0.0184168,0.015092,0.0121551,0.00962175,0.00748565,0.00572382,0.00430153,0.00317718,0.00230643,0.00164559,0.00115394,0.00079529,0.000538703,0.000358636,0.00023466,0.000150906,9.53793e-05,5.92493e-05,3.61737e-05,2.17062e-05,1.28014e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num80(theCoeff) );
	}
	{
		REAL8 theCoeff[63] = {1.56608e-05,2.61953e-05,4.30833e-05,6.96738e-05,0.000110791,0.000173227,0.000266318,0.000402588,0.000598407,0.000874595,0.00125688,0.00177604,0.00246769,0.00337133,0.00452884,0.00598202,0.00776934,0.00992192,0.012459,0.0153831,0.0186758,0.0222942,0.0261685,0.0302023,0.034275,0.0382463,0.0419639,0.0452729,0.0480259,0.0500943,0.0513778,0.0518129,0.0513778,0.0500943,0.0480259,0.0452729,0.0419639,0.0382463,0.034275,0.0302023,0.0261685,0.0222942,0.0186758,0.0153831,0.012459,0.00992192,0.00776934,0.00598202,0.00452884,0.00337133,0.00246769,0.00177604,0.00125688,0.000874595,0.000598407,0.000402588,0.000266318,0.000173227,0.000110791,6.96738e-05,4.30833e-05,2.61953e-05,1.56608e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num81(theCoeff) );
	}
	{
		REAL8 theCoeff[65] = {1.13244e-05,1.90052e-05,3.13756e-05,5.09532e-05,8.13979e-05,0.000127914,0.000197734,0.000300683,0.000449777,0.000661831,0.000957985,0.00136406,0.00191059,0.00263248,0.00356799,0.00475711,0.00623915,0.00804951,0.0102159,0.0127539,0.0156628,0.0189217,0.0224861,0.0262862,0.0302276,0.0341932,0.0380486,0.0416486,0.0448459,0.0475014,0.0494939,0.0507294,0.051148,0.0507294,0.0494939,0.0475014,0.0448459,0.0416486,0.0380486,0.0341932,0.0302276,0.0262862,0.0224861,0.0189217,0.0156628,0.0127539,0.0102159,0.00804951,0.00623915,0.00475711,0.00356799,0.00263248,0.00191059,0.00136406,0.000957985,0.000661831,0.000449777,0.000300683,0.000197734,0.000127914,8.13979e-05,5.09532e-05,3.13756e-05,1.90052e-05,1.13244e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num82(theCoeff) );
	}
	{
		REAL8 theCoeff[65] = {1.38175e-05,2.28891e-05,3.73139e-05,5.98622e-05,9.45097e-05,0.000146839,0.000224516,0.000337828,0.000500247,0.000728979,0.00104541,0.00147536,0.00204905,0.00280058,0.00376691,0.00498612,0.00649504,0.00832611,0.0105037,0.0130403,0.015932,0.0191556,0.0226654,0.0263919,0.0302427,0.0341044,0.037848,0.0413348,0.0444252,0.0469878,0.0489083,0.050098,0.050501,0.050098,0.0489083,0.0469878,0.0444252,0.0413348,0.037848,0.0341044,0.0302427,0.0263919,0.0226654,0.0191556,0.015932,0.0130403,0.0105037,0.00832611,0.00649504,0.00498612,0.00376691,0.00280058,0.00204905,0.00147536,0.00104541,0.000728979,0.000500247,0.000337828,0.000224516,0.000146839,9.45097e-05,5.98622e-05,3.73139e-05,2.28891e-05,1.38175e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num83(theCoeff) );
	}
	{
		REAL8 theCoeff[65] = {1.67296e-05,2.73678e-05,4.40766e-05,6.9886e-05,0.000109091,0.000167648,0.000253643,0.0003778,0.000554008,0.000799803,0.00113675,0.0015906,0.00219114,0.00297163,0.00396765,0.00521537,0.00674919,0.0085987,0.0107852,0.0133179,0.0161905,0.0193775,0.0228322,0.0264859,0.0302478,0.0340085,0.037644,0.0410221,0.0440103,0.0464841,0.0483358,0.0494821,0.0498702,0.0494821,0.0483358,0.0464841,0.0440103,0.0410221,0.037644,0.0340085,0.0302478,0.0264859,0.0228322,0.0193775,0.0161905,0.0133179,0.0107852,0.0085987,0.00674919,0.00521537,0.00396765,0.00297163,0.00219114,0.0015906,0.00113675,0.000799803,0.000554008,0.0003778,0.000253643,0.000167648,0.000109091,6.9886e-05,4.40766e-05,2.73678e-05,1.67296e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num84(theCoeff) );
	}
	{
		REAL8 theCoeff[67] = {1.22522e-05,2.01067e-05,3.24975e-05,5.17297e-05,8.10981e-05,0.000125216,0.000190411,0.000285171,0.000420628,0.000611044,0.000874232,0.00123186,0.00170953,0.00233654,0.00314522,0.00416973,0.00544436,0.00700109,0.00886676,0.0110598,0.0135865,0.016438,0.0195871,0.0229865,0.0265679,0.0302428,0.0339052,0.0374362,0.0407098,0.0435999,0.0459889,0.0477751,0.0488799,0.0492539,0.0488799,0.0477751,0.0459889,0.0435999,0.0407098,0.0374362,0.0339052,0.0302428,0.0265679,0.0229865,0.0195871,0.016438,0.0135865,0.0110598,0.00886676,0.00700109,0.00544436,0.00416973,0.00314522,0.00233654,0.00170953,0.00123186,0.000874232,0.000611044,0.000420628,0.000285171,0.000190411,0.000125216,8.10981e-05,5.17297e-05,3.24975e-05,2.01067e-05,1.22522e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num85(theCoeff) );
	}
	{
		REAL8 theCoeff[67] = {1.47999e-05,2.39979e-05,3.83378e-05,6.03425e-05,9.3575e-05,0.000142968,0.000215207,0.000319166,0.000466356,0.000671367,0.000952234,0.00133066,0.00183204,0.00248508,0.00332115,0.00437299,0.00567294,0.0072507,0.00913045,0.0113278,0.0138465,0.0166754,0.0197858,0.0231298,0.0266398,0.0302295,0.0337966,0.0372268,0.0403998,0.0431961,0.0455041,0.0472278,0.0482932,0.0486536,0.0482932,0.0472278,0.0455041,0.0431961,0.0403998,0.0372268,0.0337966,0.0302295,0.0266398,0.0231298,0.0197858,0.0166754,0.0138465,0.0113278,0.00913045,0.0072507,0.00567294,0.00437299,0.00332115,0.00248508,0.00183204,0.00133066,0.000952234,0.000671367,0.000466356,0.000319166,0.000215207,0.000142968,9.3575e-05,6.03425e-05,3.83378e-05,2.39979e-05,1.47999e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num86(theCoeff) );
	}
	{
		REAL8 theCoeff[69] = {1.09153e-05,1.7751e-05,2.84518e-05,4.4946e-05,6.9979e-05,0.000107384,0.000162409,0.000242089,0.00035566,0.000514982,0.000734927,0.00103369,0.00143297,0.00195783,0.0026364,0.00349899,0.00457688,0.00590055,0.00749742,0.00938915,0.0115888,0.0140975,0.0169023,0.019973,0.0232615,0.026701,0.0302073,0.0336817,0.0370144,0.0400907,0.0427969,0.0450274,0.0466915,0.0477193,0.0480669,0.0477193,0.0466915,0.0450274,0.0427969,0.0400907,0.0370144,0.0336817,0.0302073,0.026701,0.0232615,0.019973,0.0169023,0.0140975,0.0115888,0.00938915,0.00749742,0.00590055,0.00457688,0.00349899,0.0026364,0.00195783,0.00143297,0.00103369,0.000734927,0.000514982,0.00035566,0.000242089,0.000162409,0.000107384,6.9979e-05,4.4946e-05,2.84518e-05,1.7751e-05,1.09153e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num87(theCoeff) );
	}
	{
		REAL8 theCoeff[69] = {1.31546e-05,2.1148e-05,3.35201e-05,5.23825e-05,8.07073e-05,0.000122598,0.000183612,0.000271121,0.000394702,0.000566527,0.00080171,0.00111856,0.00153867,0.00208679,0.00279032,0.00367855,0.00478127,0.0061271,0.00774127,0.00964304,0.011843,0.0143401,0.0171195,0.0201499,0.023383,0.026753,0.0301779,0.0335623,0.0368009,0.0397842,0.0424041,0.0445606,0.0461677,0.0471596,0.047495,0.0471596,0.0461677,0.0445606,0.0424041,0.0397842,0.0368009,0.0335623,0.0301779,0.026753,0.023383,0.0201499,0.0171195,0.0143401,0.011843,0.00964304,0.00774127,0.0061271,0.00478127,0.00367855,0.00279032,0.00208679,0.00153867,0.00111856,0.00080171,0.000566527,0.000394702,0.000271121,0.000183612,0.000122598,8.07073e-05,5.23825e-05,3.35201e-05,2.1148e-05,1.31546e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num88(theCoeff) );
	}
	{
		REAL8 theCoeff[69] = {1.57455e-05,2.50337e-05,3.92538e-05,6.07056e-05,9.259e-05,0.00013928,0.000206634,0.000302346,0.00043631,0.000620978,0.000871657,0.00120671,0.0016476,0.00221864,0.00294655,0.00385947,0.00498576,0.00635219,0.00798187,0.0098918,0.0120902,0.0145741,0.0173269,0.0203164,0.0234943,0.0267958,0.0301412,0.0334382,0.036586,0.0394798,0.0420169,0.0441025,0.0456552,0.046613,0.0469367,0.046613,0.0456552,0.0441025,0.0420169,0.0394798,0.036586,0.0334382,0.0301412,0.0267958,0.0234943,0.0203164,0.0173269,0.0145741,0.0120902,0.0098918,0.00798187,0.00635219,0.00498576,0.00385947,0.00294655,0.00221864,0.0016476,0.00120671,0.000871657,0.000620978,0.00043631,0.000302346,0.000206634,0.00013928,9.259e-05,6.07056e-05,3.92538e-05,2.50337e-05,1.57455e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num89(theCoeff) );
	}
	{
		REAL8 theCoeff[71] = {1.1744e-05,1.87241e-05,2.94519e-05,4.5704e-05,6.99717e-05,0.000105686,0.000157487,0.000231524,0.000335797,0.000480491,0.0006783,0.000944683,0.00129801,0.00175954,0.00235314,0.00310473,0.00404136,0.00518992,0.00657538,0.00821881,0.010135,0.0123302,0.0147993,0.0175244,0.0204725,0.0235954,0.0268294,0.030097,0.0333091,0.036369,0.0391766,0.0416343,0.0436519,0.0451527,0.0460778,0.0463903,0.0460778,0.0451527,0.0436519,0.0416343,0.0391766,0.036369,0.0333091,0.030097,0.0268294,0.0235954,0.0204725,0.0175244,0.0147993,0.0123302,0.010135,0.00821881,0.00657538,0.00518992,0.00404136,0.00310473,0.00235314,0.00175954,0.00129801,0.000944683,0.0006783,0.000480491,0.000335797,0.000231524,0.000157487,0.000105686,6.99717e-05,4.5704e-05,2.94519e-05,1.87241e-05,1.1744e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num90(theCoeff) );
	}
	{
		REAL8 theCoeff[71] = {1.40283e-05,2.21288e-05,3.44489e-05,5.29241e-05,8.02407e-05,0.00012006,0.000177281,0.000258339,0.000371518,0.000527269,0.000738492,0.00102076,0.00139239,0.0018744,0.00249014,0.00326474,0.00422412,0.00539369,0.00679669,0.00845223,0.0103731,0.0125633,0.0150164,0.0177128,0.0206193,0.0236875,0.0268552,0.0300469,0.0331768,0.0361518,0.0388765,0.041258,0.0432106,0.0446616,0.0455555,0.0458575,0.0455555,0.0446616,0.0432106,0.041258,0.0388765,0.0361518,0.0331768,0.0300469,0.0268552,0.0236875,0.0206193,0.0177128,0.0150164,0.0125633,0.0103731,0.00845223,0.00679669,0.00539369,0.00422412,0.00326474,0.00249014,0.0018744,0.00139239,0.00102076,0.000738492,0.000527269,0.000371518,0.000258339,0.000177281,0.00012006,8.02407e-05,5.29241e-05,3.44489e-05,2.21288e-05,1.40283e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num91(theCoeff) );
	}
	{
		REAL8 theCoeff[73] = {1.05285e-05,1.66516e-05,2.59978e-05,4.0069e-05,6.0964e-05,9.15651e-05,0.000135762,0.000198709,0.000287111,0.000409519,0.00057662,0.000801489,0.00109976,0.00148966,0.00199192,0.00262934,0.00342621,0.0044073,0.00559658,0.00701561,0.0086816,0.0106054,0.0127892,0.0152249,0.0178919,0.0207563,0.0237703,0.0268727,0.0299903,0.0330402,0.0359331,0.038578,0.0408862,0.0427765,0.04418,0.0450441,0.0453359,0.0450441,0.04418,0.0427765,0.0408862,0.038578,0.0359331,0.0330402,0.0299903,0.0268727,0.0237703,0.0207563,0.0178919,0.0152249,0.0127892,0.0106054,0.0086816,0.00701561,0.00559658,0.0044073,0.00342621,0.00262934,0.00199192,0.00148966,0.00109976,0.000801489,0.00057662,0.000409519,0.000287111,0.000198709,0.000135762,9.15651e-05,6.0964e-05,4.0069e-05,2.59978e-05,1.66516e-05,1.05285e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num92(theCoeff) );
	}
	{
		REAL8 theCoeff[73] = {1.25507e-05,1.96475e-05,3.03715e-05,4.63596e-05,6.98766e-05,0.000104002,0.000152851,0.000221825,0.000317886,0.00044983,0.000628555,0.000867272,0.00118164,0.00158976,0.002112,0.00277061,0.003589,0.0045908,0.00579856,0.00723219,0.0089071,0.0108323,0.0130083,0.0154255,0.0180624,0.0208846,0.023845,0.0268833,0.0299287,0.032901,0.0357147,0.0382827,0.0405205,0.0423511,0.0437091,0.0445447,0.0448268,0.0445447,0.0437091,0.0423511,0.0405205,0.0382827,0.0357147,0.032901,0.0299287,0.0268833,0.023845,0.0208846,0.0180624,0.0154255,0.0130083,0.0108323,0.0089071,0.00723219,0.00579856,0.0045908,0.003589,0.00277061,0.002112,0.00158976,0.00118164,0.000867272,0.000628555,0.00044983,0.000317886,0.000221825,0.000152851,0.000104002,6.98766e-05,4.63596e-05,3.03715e-05,1.96475e-05,1.25507e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num93(theCoeff) );
	}
	{
		REAL8 theCoeff[73] = {1.48708e-05,2.30501e-05,3.52898e-05,5.33661e-05,7.97111e-05,0.000117601,0.000171373,0.000246667,0.000350686,0.000492452,0.000683043,0.000935772,0.00126628,0.0016925,0.00223443,0.00291369,0.00375281,0.0047743,0.00599929,0.00744611,0.00912844,0.0110536,0.0132205,0.0156181,0.0182242,0.0210043,0.0239115,0.026887,0.0298618,0.0327588,0.035496,0.0379899,0.0401602,0.0419335,0.043248,0.0440563,0.0443291,0.0440563,0.043248,0.0419335,0.0401602,0.0379899,0.035496,0.0327588,0.0298618,0.026887,0.0239115,0.0210043,0.0182242,0.0156181,0.0132205,0.0110536,0.00912844,0.00744611,0.00599929,0.0047743,0.00375281,0.00291369,0.00223443,0.0016925,0.00126628,0.000935772,0.000683043,0.000492452,0.000350686,0.000246667,0.000171373,0.000117601,7.97111e-05,5.33661e-05,3.52898e-05,2.30501e-05,1.48708e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num94(theCoeff) );
	}
	{
		REAL8 theCoeff[75] = {1.12732e-05,1.75174e-05,2.68935e-05,4.07926e-05,6.11324e-05,9.05142e-05,0.000132409,0.00019137,0.000273266,0.000385525,0.000537373,0.000740039,0.0010069,0.00135356,0.00179772,0.00235898,0.0030583,0.00391734,0.00495745,0.00619842,0.00765701,0.00934529,0.0112689,0.0134254,0.0158026,0.0183774,0.0211152,0.0239696,0.0268834,0.0297894,0.0326134,0.0352764,0.0376989,0.0398042,0.0415226,0.0427953,0.0435775,0.0438415,0.0435775,0.0427953,0.0415226,0.0398042,0.0376989,0.0352764,0.0326134,0.0297894,0.0268834,0.0239696,0.0211152,0.0183774,0.0158026,0.0134254,0.0112689,0.00934529,0.00765701,0.00619842,0.00495745,0.00391734,0.0030583,0.00235898,0.00179772,0.00135356,0.0010069,0.000740039,0.000537373,0.000385525,0.000273266,0.00019137,0.000132409,9.05142e-05,6.11324e-05,4.07926e-05,2.68935e-05,1.75174e-05,1.12732e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num95(theCoeff) );
	}
	{
		REAL8 theCoeff[75] = {1.33327e-05,2.05212e-05,3.12145e-05,4.69222e-05,6.97059e-05,0.000102336,0.000148477,0.000212891,0.000301664,0.000422434,0.000584607,0.000799535,0.00108064,0.00144342,0.00190534,0.00248554,0.00320434,0.0040825,0.00514023,0.00639598,0.00786504,0.00955793,0.0114788,0.0136237,0.0159796,0.0185227,0.0212184,0.0240208,0.0268741,0.0297131,0.0324661,0.0350576,0.0374114,0.0394542,0.0411199,0.0423526,0.0431098,0.0433653,0.0431098,0.0423526,0.0411199,0.0394542,0.0374114,0.0350576,0.0324661,0.0297131,0.0268741,0.0240208,0.0212184,0.0185227,0.0159796,0.0136237,0.0114788,0.00955793,0.00786504,0.00639598,0.00514023,0.0040825,0.00320434,0.00248554,0.00190534,0.00144342,0.00108064,0.000799535,0.000584607,0.000422434,0.000301664,0.000212891,0.000148477,0.000102336,6.97059e-05,4.69222e-05,3.12145e-05,2.05212e-05,1.33327e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num96(theCoeff) );
	}
	{
		REAL8 theCoeff[77] = {1.01635e-05,1.56799e-05,2.39124e-05,3.60479e-05,5.37175e-05,7.9128e-05,0.000115219,0.000165842,0.000235964,0.000331875,0.000461405,0.000634116,0.000861458,0.00115685,0.00153568,0.00201512,0.00261385,0.0033515,0.00424793,0.00532222,0.00659155,0.00806976,0.0097659,0.0116827,0.013815,0.0161488,0.0186598,0.0213134,0.0240644,0.0268583,0.0296319,0.0323161,0.0348384,0.0371258,0.0391085,0.0407236,0.0419179,0.0426512,0.0428985,0.0426512,0.0419179,0.0407236,0.0391085,0.0371258,0.0348384,0.0323161,0.0296319,0.0268583,0.0240644,0.0213134,0.0186598,0.0161488,0.013815,0.0116827,0.0097659,0.00806976,0.00659155,0.00532222,0.00424793,0.0033515,0.00261385,0.00201512,0.00153568,0.00115685,0.000861458,0.000634116,0.000461405,0.000331875,0.000235964,0.000165842,0.000115219,7.9128e-05,5.37175e-05,3.60479e-05,2.39124e-05,1.56799e-05,1.01635e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num97(theCoeff) );
	}
	{
		REAL8 theCoeff[77] = {1.19986e-05,1.83419e-05,2.77232e-05,4.14312e-05,6.12205e-05,8.94441e-05,0.000129208,0.00018455,0.00026063,0.000363931,0.000502456,0.000685902,0.000925787,0.00123551,0.00163029,0.002127,0.00274383,0.00349969,0.00441356,0.00550342,0.00678517,0.00827131,0.00996949,0.0118811,0.0139999,0.0163109,0.0187895,0.0214013,0.0241017,0.0268374,0.0295473,0.0321648,0.0346202,0.0368436,0.0387686,0.040335,0.0414926,0.0422029,0.0424424,0.0422029,0.0414926,0.040335,0.0387686,0.0368436,0.0346202,0.0321648,0.0295473,0.0268374,0.0241017,0.0214013,0.0187895,0.0163109,0.0139999,0.0118811,0.00996949,0.00827131,0.00678517,0.00550342,0.00441356,0.00349969,0.00274383,0.002127,0.00163029,0.00123551,0.000925787,0.000685902,0.000502456,0.000363931,0.00026063,0.00018455,0.000129208,8.94441e-05,6.12205e-05,4.14312e-05,2.77232e-05,1.83419e-05,1.19986e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num98(theCoeff) );
	}
	{
		REAL8 theCoeff[77] = {1.40881e-05,2.13455e-05,3.19853e-05,4.74002e-05,6.94702e-05,0.000100694,0.000144344,0.000204636,0.000286914,0.00039784,0.000545575,0.000739925,0.000992451,0.00131649,0.00172709,0.00224079,0.00287524,0.00364867,0.00457913,0.00568355,0.0069766,0.00846946,0.0101685,0.0120738,0.0141782,0.0164658,0.0189119,0.021482,0.0241325,0.0268113,0.0294591,0.0320119,0.0344025,0.0365643,0.0384336,0.0399534,0.0410756,0.041764,0.041996,0.041764,0.0410756,0.0399534,0.0384336,0.0365643,0.0344025,0.0320119,0.0294591,0.0268113,0.0241325,0.021482,0.0189119,0.0164658,0.0141782,0.0120738,0.0101685,0.00846946,0.0069766,0.00568355,0.00457913,0.00364867,0.00287524,0.00224079,0.00172709,0.00131649,0.000992451,0.000739925,0.000545575,0.00039784,0.000286914,0.000204636,0.000144344,0.000100694,6.94702e-05,4.74002e-05,3.19853e-05,2.13455e-05,1.40881e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num99(theCoeff) );
	}
	{
		REAL8 theCoeff[79] = {1.08362e-05,1.64553e-05,2.47184e-05,3.67301e-05,5.39899e-05,7.85037e-05,0.000112916,0.00016066,0.000226125,0.000314831,0.000433603,0.00059074,0.000796135,0.00106137,0.00139969,0.00182593,0.00235628,0.00300785,0.00379815,0.00474434,0.0058623,0.00716551,0.00866392,0.0103626,0.0122606,0.0143496,0.0166134,0.0190267,0.0215554,0.0241566,0.0267796,0.029367,0.0318568,0.0341847,0.036287,0.0381028,0.0395776,0.0406659,0.0413332,0.0415581,0.0413332,0.0406659,0.0395776,0.0381028,0.036287,0.0341847,0.0318568,0.029367,0.0267796,0.0241566,0.0215554,0.0190267,0.0166134,0.0143496,0.0122606,0.0103626,0.00866392,0.00716551,0.0058623,0.00474434,0.00379815,0.00300785,0.00235628,0.00182593,0.00139969,0.00106137,0.000796135,0.00059074,0.000433603,0.000314831,0.000226125,0.00016066,0.000112916,7.85037e-05,5.39899e-05,3.67301e-05,2.47184e-05,1.64553e-05,1.08362e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num100(theCoeff) );
	}
	{
		REAL8 theCoeff[79] = {1.27026e-05,1.91248e-05,2.84897e-05,4.19916e-05,6.1238e-05,8.83616e-05,0.000126151,0.000178198,0.000249056,0.000344412,0.00047124,0.000637955,0.000854522,0.0011325,0.00148505,0.00192676,0.00247341,0.0031416,0.00394811,0.00490921,0.00603974,0.00735207,0.00885492,0.0105522,0.0124419,0.014515,0.0167544,0.0191349,0.0216225,0.0241753,0.0267437,0.0292722,0.031701,0.0339684,0.0360132,0.0377774,0.0392092,0.0402649,0.040912,0.04113,0.040912,0.0402649,0.0392092,0.0377774,0.0360132,0.0339684,0.031701,0.0292722,0.0267437,0.0241753,0.0216225,0.0191349,0.0167544,0.014515,0.0124419,0.0105522,0.00885492,0.00735207,0.00603974,0.00490921,0.00394811,0.0031416,0.00247341,0.00192676,0.00148505,0.0011325,0.000854522,0.000637955,0.00047124,0.000344412,0.000249056,0.000178198,0.000126151,8.83616e-05,6.1238e-05,4.19916e-05,2.84897e-05,1.91248e-05,1.27026e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num101(theCoeff) );
	}
	{
		REAL8 theCoeff[81] = {9.81943e-06,1.48151e-05,2.21209e-05,3.26873e-05,4.78005e-05,6.91774e-05,9.90772e-05,0.00014043,0.000196982,0.000273445,0.000375657,0.00051073,0.000687177,0.000915007,0.00120575,0.00157242,0.00202936,0.00259195,0.00327621,0.00409822,0.00507336,0.00621549,0.00753586,0.00904207,0.010737,0.0126175,0.0146738,0.0168884,0.019236,0.0216829,0.0241879,0.0267029,0.0291739,0.0315435,0.0337523,0.0357416,0.0374562,0.0388464,0.0398708,0.0404985,0.0407098,0.0404985,0.0398708,0.0388464,0.0374562,0.0357416,0.0337523,0.0315435,0.0291739,0.0267029,0.0241879,0.0216829,0.019236,0.0168884,0.0146738,0.0126175,0.010737,0.00904207,0.00753586,0.00621549,0.00507336,0.00409822,0.00327621,0.00259195,0.00202936,0.00157242,0.00120575,0.000915007,0.000687177,0.00051073,0.000375657,0.000273445,0.000196982,0.00014043,9.90772e-05,6.91774e-05,4.78005e-05,3.26873e-05,2.21209e-05,1.48151e-05,9.81943e-06};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num102(theCoeff) );
	}
	{
		REAL8 theCoeff[81] = {1.1492e-05,1.71958e-05,2.54696e-05,3.73413e-05,5.41907e-05,7.78448e-05,0.000110689,0.000155792,0.000217049,0.000299322,0.00040859,0.000552085,0.000738402,0.000977572,0.00128107,0.00166176,0.00213368,0.00271182,0.00341163,0.00424845,0.00523683,0.00638963,0.00771705,0.00922563,0.0109172,0.0127877,0.0148267,0.0170162,0.0193309,0.0217375,0.0241956,0.0266583,0.0290735,0.0313856,0.0335377,0.0354735,0.0371402,0.0384905,0.0394849,0.0400939,0.0402989,0.0400939,0.0394849,0.0384905,0.0371402,0.0354735,0.0335377,0.0313856,0.0290735,0.0266583,0.0241956,0.0217375,0.0193309,0.0170162,0.0148267,0.0127877,0.0109172,0.00922563,0.00771705,0.00638963,0.00523683,0.00424845,0.00341163,0.00271182,0.00213368,0.00166176,0.00128107,0.000977572,0.000738402,0.000552085,0.00040859,0.000299322,0.000217049,0.000155792,0.000110689,7.78448e-05,5.41907e-05,3.73413e-05,2.54696e-05,1.71958e-05,1.1492e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num103(theCoeff) );
	}
	{
		REAL8 theCoeff[81] = {1.33837e-05,1.98666e-05,2.91962e-05,4.24802e-05,6.11933e-05,8.72727e-05,0.000123228,0.000172266,0.000238421,0.000326699,0.000443207,0.000595284,0.000791585,0.00104215,0.00135837,0.00175292,0.00223957,0.00283285,0.00354764,0.00439858,0.00539937,0.00656192,0.00789542,0.00940539,0.0110926,0.0129524,0.0149735,0.0171377,0.0194196,0.0217863,0.0241983,0.0266099,0.0289706,0.031227,0.0333242,0.0352083,0.0368289,0.0381407,0.0391063,0.0396973,0.0398963,0.0396973,0.0391063,0.0381407,0.0368289,0.0352083,0.0333242,0.031227,0.0289706,0.0266099,0.0241983,0.0217863,0.0194196,0.0171377,0.0149735,0.0129524,0.0110926,0.00940539,0.00789542,0.00656192,0.00539937,0.00439858,0.00354764,0.00283285,0.00223957,0.00175292,0.00135837,0.00104215,0.000791585,0.000595284,0.000443207,0.000326699,0.000238421,0.000172266,0.000123228,8.72727e-05,6.11933e-05,4.24802e-05,2.91962e-05,1.98666e-05,1.33837e-05};
		mConvolutions.push_back( new cConvolSpec_REAL4_Num104(theCoeff) );
	}
}
