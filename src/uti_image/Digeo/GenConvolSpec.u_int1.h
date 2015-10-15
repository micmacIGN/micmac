#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num0 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num0(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   32768*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num0(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(0),0,0,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num1 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num1(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   126*(INT(In[-1])+INT(In[1]))
				              +   32516*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num1(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-1),-1,1,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num2 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num2(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1323*(INT(In[-1])+INT(In[1]))
				              +   30122*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num2(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-1),-1,1,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num3 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num3(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   9*(INT(In[-2])+INT(In[2]))
				              +   3488*(INT(In[-1])+INT(In[1]))
				              +   25774*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num3(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-2),-2,2,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num4 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num4(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   84*(INT(In[-2])+INT(In[2]))
				              +   5424*(INT(In[-1])+INT(In[1]))
				              +   21752*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num4(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-2),-2,2,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num5 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num5(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-3])+INT(In[3]))
				              +   315*(INT(In[-2])+INT(In[2]))
				              +   6731*(INT(In[-1])+INT(In[1]))
				              +   18672*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num5(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-3),-3,3,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num6 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num6(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   14*(INT(In[-3])+INT(In[3]))
				              +   718*(INT(In[-2])+INT(In[2]))
				              +   7481*(INT(In[-1])+INT(In[1]))
				              +   16342*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num6(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-3),-3,3,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num7 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num7(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-4])+INT(In[4]))
				              +   56*(INT(In[-3])+INT(In[3]))
				              +   1230*(INT(In[-2])+INT(In[2]))
				              +   7835*(INT(In[-1])+INT(In[1]))
				              +   14524*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num7(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-4),-4,4,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num8 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num8(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4*(INT(In[-4])+INT(In[4]))
				              +   145*(INT(In[-3])+INT(In[3]))
				              +   1769*(INT(In[-2])+INT(In[2]))
				              +   7929*(INT(In[-1])+INT(In[1]))
				              +   13074*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num8(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-4),-4,4,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num9 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num9(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   16*(INT(In[-4])+INT(In[4]))
				              +   288*(INT(In[-3])+INT(In[3]))
				              +   2276*(INT(In[-2])+INT(In[2]))
				              +   7862*(INT(In[-1])+INT(In[1]))
				              +   11884*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num9(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-4),-4,4,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num10 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num10(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-5])+INT(In[5]))
				              +   42*(INT(In[-4])+INT(In[4]))
				              +   479*(INT(In[-3])+INT(In[3]))
				              +   2716*(INT(In[-2])+INT(In[2]))
				              +   7698*(INT(In[-1])+INT(In[1]))
				              +   10894*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num10(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-5),-5,5,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num11 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num11(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   6*(INT(In[-5])+INT(In[5]))
				              +   88*(INT(In[-4])+INT(In[4]))
				              +   702*(INT(In[-3])+INT(In[3]))
				              +   3079*(INT(In[-2])+INT(In[2]))
				              +   7481*(INT(In[-1])+INT(In[1]))
				              +   10056*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num11(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-5),-5,5,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num12 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num12(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-6])+INT(In[6]))
				              +   16*(INT(In[-5])+INT(In[5]))
				              +   158*(INT(In[-4])+INT(In[4]))
				              +   940*(INT(In[-3])+INT(In[3]))
				              +   3366*(INT(In[-2])+INT(In[2]))
				              +   7235*(INT(In[-1])+INT(In[1]))
				              +   9336*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num12(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-6),-6,6,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num13 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num13(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   3*(INT(In[-6])+INT(In[6]))
				              +   34*(INT(In[-5])+INT(In[5]))
				              +   249*(INT(In[-4])+INT(In[4]))
				              +   1179*(INT(In[-3])+INT(In[3]))
				              +   3583*(INT(In[-2])+INT(In[2]))
				              +   6978*(INT(In[-1])+INT(In[1]))
				              +   8716*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num13(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-6),-6,6,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num14 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num14(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   7*(INT(In[-6])+INT(In[6]))
				              +   62*(INT(In[-5])+INT(In[5]))
				              +   359*(INT(In[-4])+INT(In[4]))
				              +   1409*(INT(In[-3])+INT(In[3]))
				              +   3741*(INT(In[-2])+INT(In[2]))
				              +   6721*(INT(In[-1])+INT(In[1]))
				              +   8170*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num14(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-6),-6,6,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num15 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num15(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-7])+INT(In[7]))
				              +   15*(INT(In[-6])+INT(In[6]))
				              +   102*(INT(In[-5])+INT(In[5]))
				              +   483*(INT(In[-4])+INT(In[4]))
				              +   1621*(INT(In[-3])+INT(In[3]))
				              +   3849*(INT(In[-2])+INT(In[2]))
				              +   6468*(INT(In[-1])+INT(In[1]))
				              +   7690*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num15(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-7),-7,7,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num16 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num16(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   4*(INT(In[-7])+INT(In[7]))
				              +   28*(INT(In[-6])+INT(In[6]))
				              +   153*(INT(In[-5])+INT(In[5]))
				              +   615*(INT(In[-4])+INT(In[4]))
				              +   1811*(INT(In[-3])+INT(In[3]))
				              +   3917*(INT(In[-2])+INT(In[2]))
				              +   6224*(INT(In[-1])+INT(In[1]))
				              +   7264*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num16(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-7),-7,7,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num17 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num17(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-8])+INT(In[8]))
				              +   8*(INT(In[-7])+INT(In[7]))
				              +   47*(INT(In[-6])+INT(In[6]))
				              +   216*(INT(In[-5])+INT(In[5]))
				              +   750*(INT(In[-4])+INT(In[4]))
				              +   1978*(INT(In[-3])+INT(In[3]))
				              +   3954*(INT(In[-2])+INT(In[2]))
				              +   5990*(INT(In[-1])+INT(In[1]))
				              +   6880*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num17(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-8),-8,8,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num18 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num18(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-8])+INT(In[8]))
				              +   14*(INT(In[-7])+INT(In[7]))
				              +   73*(INT(In[-6])+INT(In[6]))
				              +   287*(INT(In[-5])+INT(In[5]))
				              +   885*(INT(In[-4])+INT(In[4]))
				              +   2122*(INT(In[-3])+INT(In[3]))
				              +   3965*(INT(In[-2])+INT(In[2]))
				              +   5768*(INT(In[-1])+INT(In[1]))
				              +   6536*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num18(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-8),-8,8,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num19 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num19(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-9])+INT(In[9]))
				              +   4*(INT(In[-8])+INT(In[8]))
				              +   24*(INT(In[-7])+INT(In[7]))
				              +   105*(INT(In[-6])+INT(In[6]))
				              +   366*(INT(In[-5])+INT(In[5]))
				              +   1015*(INT(In[-4])+INT(In[4]))
				              +   2244*(INT(In[-3])+INT(In[3]))
				              +   3955*(INT(In[-2])+INT(In[2]))
				              +   5558*(INT(In[-1])+INT(In[1]))
				              +   6224*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num19(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-9),-9,9,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num20 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num20(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-9])+INT(In[9]))
				              +   8*(INT(In[-8])+INT(In[8]))
				              +   38*(INT(In[-7])+INT(In[7]))
				              +   144*(INT(In[-6])+INT(In[6]))
				              +   449*(INT(In[-5])+INT(In[5]))
				              +   1138*(INT(In[-4])+INT(In[4]))
				              +   2345*(INT(In[-3])+INT(In[3]))
				              +   3931*(INT(In[-2])+INT(In[2]))
				              +   5359*(INT(In[-1])+INT(In[1]))
				              +   5942*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num20(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-9),-9,9,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num21 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num21(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-10])+INT(In[10]))
				              +   3*(INT(In[-9])+INT(In[9]))
				              +   13*(INT(In[-8])+INT(In[8]))
				              +   55*(INT(In[-7])+INT(In[7]))
				              +   189*(INT(In[-6])+INT(In[6]))
				              +   535*(INT(In[-5])+INT(In[5]))
				              +   1253*(INT(In[-4])+INT(In[4]))
				              +   2428*(INT(In[-3])+INT(In[3]))
				              +   3894*(INT(In[-2])+INT(In[2]))
				              +   5171*(INT(In[-1])+INT(In[1]))
				              +   5684*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num21(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-10),-10,10,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num22 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num22(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-10])+INT(In[10]))
				              +   5*(INT(In[-9])+INT(In[9]))
				              +   21*(INT(In[-8])+INT(In[8]))
				              +   77*(INT(In[-7])+INT(In[7]))
				              +   239*(INT(In[-6])+INT(In[6]))
				              +   622*(INT(In[-5])+INT(In[5]))
				              +   1358*(INT(In[-4])+INT(In[4]))
				              +   2494*(INT(In[-3])+INT(In[3]))
				              +   3849*(INT(In[-2])+INT(In[2]))
				              +   4994*(INT(In[-1])+INT(In[1]))
				              +   5448*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num22(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-10),-10,10,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num23 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num23(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-10])+INT(In[10]))
				              +   8*(INT(In[-9])+INT(In[9]))
				              +   31*(INT(In[-8])+INT(In[8]))
				              +   104*(INT(In[-7])+INT(In[7]))
				              +   294*(INT(In[-6])+INT(In[6]))
				              +   708*(INT(In[-5])+INT(In[5]))
				              +   1454*(INT(In[-4])+INT(In[4]))
				              +   2545*(INT(In[-3])+INT(In[3]))
				              +   3797*(INT(In[-2])+INT(In[2]))
				              +   4827*(INT(In[-1])+INT(In[1]))
				              +   5228*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num23(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-10),-10,10,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num24 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num24(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-11])+INT(In[11]))
				              +   3*(INT(In[-10])+INT(In[10]))
				              +   13*(INT(In[-9])+INT(In[9]))
				              +   44*(INT(In[-8])+INT(In[8]))
				              +   134*(INT(In[-7])+INT(In[7]))
				              +   351*(INT(In[-6])+INT(In[6]))
				              +   791*(INT(In[-5])+INT(In[5]))
				              +   1540*(INT(In[-4])+INT(In[4]))
				              +   2584*(INT(In[-3])+INT(In[3]))
				              +   3740*(INT(In[-2])+INT(In[2]))
				              +   4669*(INT(In[-1])+INT(In[1]))
				              +   5028*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num24(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-11),-11,11,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num25 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num25(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-11])+INT(In[11]))
				              +   5*(INT(In[-10])+INT(In[10]))
				              +   19*(INT(In[-9])+INT(In[9]))
				              +   60*(INT(In[-8])+INT(In[8]))
				              +   168*(INT(In[-7])+INT(In[7]))
				              +   410*(INT(In[-6])+INT(In[6]))
				              +   871*(INT(In[-5])+INT(In[5]))
				              +   1616*(INT(In[-4])+INT(In[4]))
				              +   2612*(INT(In[-3])+INT(In[3]))
				              +   3680*(INT(In[-2])+INT(In[2]))
				              +   4521*(INT(In[-1])+INT(In[1]))
				              +   4842*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num25(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-11),-11,11,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num26 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num26(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-11])+INT(In[11]))
				              +   8*(INT(In[-10])+INT(In[10]))
				              +   27*(INT(In[-9])+INT(In[9]))
				              +   79*(INT(In[-8])+INT(In[8]))
				              +   205*(INT(In[-7])+INT(In[7]))
				              +   470*(INT(In[-6])+INT(In[6]))
				              +   948*(INT(In[-5])+INT(In[5]))
				              +   1683*(INT(In[-4])+INT(In[4]))
				              +   2630*(INT(In[-3])+INT(In[3]))
				              +   3618*(INT(In[-2])+INT(In[2]))
				              +   4380*(INT(In[-1])+INT(In[1]))
				              +   4668*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num26(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-11),-11,11,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num27 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num27(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-12])+INT(In[12]))
				              +   3*(INT(In[-11])+INT(In[11]))
				              +   12*(INT(In[-10])+INT(In[10]))
				              +   36*(INT(In[-9])+INT(In[9]))
				              +   100*(INT(In[-8])+INT(In[8]))
				              +   245*(INT(In[-7])+INT(In[7]))
				              +   530*(INT(In[-6])+INT(In[6]))
				              +   1020*(INT(In[-5])+INT(In[5]))
				              +   1741*(INT(In[-4])+INT(In[4]))
				              +   2640*(INT(In[-3])+INT(In[3]))
				              +   3554*(INT(In[-2])+INT(In[2]))
				              +   4248*(INT(In[-1])+INT(In[1]))
				              +   4508*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num27(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-12),-12,12,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num28 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num28(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-12])+INT(In[12]))
				              +   5*(INT(In[-11])+INT(In[11]))
				              +   17*(INT(In[-10])+INT(In[10]))
				              +   48*(INT(In[-9])+INT(In[9]))
				              +   125*(INT(In[-8])+INT(In[8]))
				              +   286*(INT(In[-7])+INT(In[7]))
				              +   590*(INT(In[-6])+INT(In[6]))
				              +   1087*(INT(In[-5])+INT(In[5]))
				              +   1791*(INT(In[-4])+INT(In[4]))
				              +   2643*(INT(In[-3])+INT(In[3]))
				              +   3489*(INT(In[-2])+INT(In[2]))
				              +   4122*(INT(In[-1])+INT(In[1]))
				              +   4358*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num28(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-12),-12,12,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num29 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num29(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-13])+INT(In[13]))
				              +   2*(INT(In[-12])+INT(In[12]))
				              +   8*(INT(In[-11])+INT(In[11]))
				              +   23*(INT(In[-10])+INT(In[10]))
				              +   62*(INT(In[-9])+INT(In[9]))
				              +   151*(INT(In[-8])+INT(In[8]))
				              +   330*(INT(In[-7])+INT(In[7]))
				              +   648*(INT(In[-6])+INT(In[6]))
				              +   1148*(INT(In[-5])+INT(In[5]))
				              +   1834*(INT(In[-4])+INT(In[4]))
				              +   2640*(INT(In[-3])+INT(In[3]))
				              +   3425*(INT(In[-2])+INT(In[2]))
				              +   4003*(INT(In[-1])+INT(In[1]))
				              +   4218*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num29(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-13),-13,13,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num30 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num30(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-13])+INT(In[13]))
				              +   4*(INT(In[-12])+INT(In[12]))
				              +   11*(INT(In[-11])+INT(In[11]))
				              +   31*(INT(In[-10])+INT(In[10]))
				              +   78*(INT(In[-9])+INT(In[9]))
				              +   180*(INT(In[-8])+INT(In[8]))
				              +   373*(INT(In[-7])+INT(In[7]))
				              +   704*(INT(In[-6])+INT(In[6]))
				              +   1205*(INT(In[-5])+INT(In[5]))
				              +   1870*(INT(In[-4])+INT(In[4]))
				              +   2633*(INT(In[-3])+INT(In[3]))
				              +   3360*(INT(In[-2])+INT(In[2]))
				              +   3891*(INT(In[-1])+INT(In[1]))
				              +   4086*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num30(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-13),-13,13,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num31 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num31(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   2*(INT(In[-13])+INT(In[13]))
				              +   5*(INT(In[-12])+INT(In[12]))
				              +   15*(INT(In[-11])+INT(In[11]))
				              +   40*(INT(In[-10])+INT(In[10]))
				              +   96*(INT(In[-9])+INT(In[9]))
				              +   210*(INT(In[-8])+INT(In[8]))
				              +   418*(INT(In[-7])+INT(In[7]))
				              +   759*(INT(In[-6])+INT(In[6]))
				              +   1257*(INT(In[-5])+INT(In[5]))
				              +   1900*(INT(In[-4])+INT(In[4]))
				              +   2621*(INT(In[-3])+INT(In[3]))
				              +   3297*(INT(In[-2])+INT(In[2]))
				              +   3784*(INT(In[-1])+INT(In[1]))
				              +   3960*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num31(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-13),-13,13,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num32 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num32(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-14])+INT(In[14]))
				              +   3*(INT(In[-13])+INT(In[13]))
				              +   8*(INT(In[-12])+INT(In[12]))
				              +   20*(INT(In[-11])+INT(In[11]))
				              +   51*(INT(In[-10])+INT(In[10]))
				              +   116*(INT(In[-9])+INT(In[9]))
				              +   241*(INT(In[-8])+INT(In[8]))
				              +   462*(INT(In[-7])+INT(In[7]))
				              +   810*(INT(In[-6])+INT(In[6]))
				              +   1304*(INT(In[-5])+INT(In[5]))
				              +   1925*(INT(In[-4])+INT(In[4]))
				              +   2605*(INT(In[-3])+INT(In[3]))
				              +   3234*(INT(In[-2])+INT(In[2]))
				              +   3682*(INT(In[-1])+INT(In[1]))
				              +   3844*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num32(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-14),-14,14,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num33 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num33(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-14])+INT(In[14]))
				              +   4*(INT(In[-13])+INT(In[13]))
				              +   10*(INT(In[-12])+INT(In[12]))
				              +   27*(INT(In[-11])+INT(In[11]))
				              +   63*(INT(In[-10])+INT(In[10]))
				              +   137*(INT(In[-9])+INT(In[9]))
				              +   274*(INT(In[-8])+INT(In[8]))
				              +   506*(INT(In[-7])+INT(In[7]))
				              +   859*(INT(In[-6])+INT(In[6]))
				              +   1346*(INT(In[-5])+INT(In[5]))
				              +   1944*(INT(In[-4])+INT(In[4]))
				              +   2587*(INT(In[-3])+INT(In[3]))
				              +   3172*(INT(In[-2])+INT(In[2]))
				              +   3586*(INT(In[-1])+INT(In[1]))
				              +   3736*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num33(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-14),-14,14,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num34 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num34(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-15])+INT(In[15]))
				              +   2*(INT(In[-14])+INT(In[14]))
				              +   5*(INT(In[-13])+INT(In[13]))
				              +   14*(INT(In[-12])+INT(In[12]))
				              +   34*(INT(In[-11])+INT(In[11]))
				              +   77*(INT(In[-10])+INT(In[10]))
				              +   160*(INT(In[-9])+INT(In[9]))
				              +   307*(INT(In[-8])+INT(In[8]))
				              +   548*(INT(In[-7])+INT(In[7]))
				              +   905*(INT(In[-6])+INT(In[6]))
				              +   1384*(INT(In[-5])+INT(In[5]))
				              +   1959*(INT(In[-4])+INT(In[4]))
				              +   2566*(INT(In[-3])+INT(In[3]))
				              +   3112*(INT(In[-2])+INT(In[2]))
				              +   3494*(INT(In[-1])+INT(In[1]))
				              +   3632*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num34(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-15),-15,15,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num35 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num35(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-15])+INT(In[15]))
				              +   3*(INT(In[-14])+INT(In[14]))
				              +   7*(INT(In[-13])+INT(In[13]))
				              +   18*(INT(In[-12])+INT(In[12]))
				              +   43*(INT(In[-11])+INT(In[11]))
				              +   92*(INT(In[-10])+INT(In[10]))
				              +   183*(INT(In[-9])+INT(In[9]))
				              +   341*(INT(In[-8])+INT(In[8]))
				              +   590*(INT(In[-7])+INT(In[7]))
				              +   949*(INT(In[-6])+INT(In[6]))
				              +   1418*(INT(In[-5])+INT(In[5]))
				              +   1970*(INT(In[-4])+INT(In[4]))
				              +   2543*(INT(In[-3])+INT(In[3]))
				              +   3053*(INT(In[-2])+INT(In[2]))
				              +   3406*(INT(In[-1])+INT(In[1]))
				              +   3534*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num35(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-15),-15,15,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num36 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num36(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-15])+INT(In[15]))
				              +   4*(INT(In[-14])+INT(In[14]))
				              +   10*(INT(In[-13])+INT(In[13]))
				              +   24*(INT(In[-12])+INT(In[12]))
				              +   52*(INT(In[-11])+INT(In[11]))
				              +   108*(INT(In[-10])+INT(In[10]))
				              +   208*(INT(In[-9])+INT(In[9]))
				              +   375*(INT(In[-8])+INT(In[8]))
				              +   631*(INT(In[-7])+INT(In[7]))
				              +   989*(INT(In[-6])+INT(In[6]))
				              +   1448*(INT(In[-5])+INT(In[5]))
				              +   1977*(INT(In[-4])+INT(In[4]))
				              +   2519*(INT(In[-3])+INT(In[3]))
				              +   2995*(INT(In[-2])+INT(In[2]))
				              +   3323*(INT(In[-1])+INT(In[1]))
				              +   3440*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num36(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-15),-15,15,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num37 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num37(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-16])+INT(In[16]))
				              +   2*(INT(In[-15])+INT(In[15]))
				              +   5*(INT(In[-14])+INT(In[14]))
				              +   13*(INT(In[-13])+INT(In[13]))
				              +   29*(INT(In[-12])+INT(In[12]))
				              +   63*(INT(In[-11])+INT(In[11]))
				              +   125*(INT(In[-10])+INT(In[10]))
				              +   234*(INT(In[-9])+INT(In[9]))
				              +   409*(INT(In[-8])+INT(In[8]))
				              +   669*(INT(In[-7])+INT(In[7]))
				              +   1026*(INT(In[-6])+INT(In[6]))
				              +   1474*(INT(In[-5])+INT(In[5]))
				              +   1981*(INT(In[-4])+INT(In[4]))
				              +   2494*(INT(In[-3])+INT(In[3]))
				              +   2939*(INT(In[-2])+INT(In[2]))
				              +   3244*(INT(In[-1])+INT(In[1]))
				              +   3352*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num37(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-16),-16,16,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num38 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num38(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-16])+INT(In[16]))
				              +   3*(INT(In[-15])+INT(In[15]))
				              +   7*(INT(In[-14])+INT(In[14]))
				              +   17*(INT(In[-13])+INT(In[13]))
				              +   36*(INT(In[-12])+INT(In[12]))
				              +   75*(INT(In[-11])+INT(In[11]))
				              +   144*(INT(In[-10])+INT(In[10]))
				              +   260*(INT(In[-9])+INT(In[9]))
				              +   442*(INT(In[-8])+INT(In[8]))
				              +   707*(INT(In[-7])+INT(In[7]))
				              +   1061*(INT(In[-6])+INT(In[6]))
				              +   1496*(INT(In[-5])+INT(In[5]))
				              +   1982*(INT(In[-4])+INT(In[4]))
				              +   2467*(INT(In[-3])+INT(In[3]))
				              +   2884*(INT(In[-2])+INT(In[2]))
				              +   3168*(INT(In[-1])+INT(In[1]))
				              +   3268*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num38(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-16),-16,16,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num39 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num39(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-17])+INT(In[17]))
				              +   1*(INT(In[-16])+INT(In[16]))
				              +   4*(INT(In[-15])+INT(In[15]))
				              +   9*(INT(In[-14])+INT(In[14]))
				              +   21*(INT(In[-13])+INT(In[13]))
				              +   44*(INT(In[-12])+INT(In[12]))
				              +   87*(INT(In[-11])+INT(In[11]))
				              +   163*(INT(In[-10])+INT(In[10]))
				              +   287*(INT(In[-9])+INT(In[9]))
				              +   475*(INT(In[-8])+INT(In[8]))
				              +   742*(INT(In[-7])+INT(In[7]))
				              +   1093*(INT(In[-6])+INT(In[6]))
				              +   1516*(INT(In[-5])+INT(In[5]))
				              +   1981*(INT(In[-4])+INT(In[4]))
				              +   2440*(INT(In[-3])+INT(In[3]))
				              +   2831*(INT(In[-2])+INT(In[2]))
				              +   3095*(INT(In[-1])+INT(In[1]))
				              +   3188*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num39(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-17),-17,17,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num40 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num40(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-17])+INT(In[17]))
				              +   2*(INT(In[-16])+INT(In[16]))
				              +   5*(INT(In[-15])+INT(In[15]))
				              +   12*(INT(In[-14])+INT(In[14]))
				              +   26*(INT(In[-13])+INT(In[13]))
				              +   53*(INT(In[-12])+INT(In[12]))
				              +   101*(INT(In[-11])+INT(In[11]))
				              +   183*(INT(In[-10])+INT(In[10]))
				              +   313*(INT(In[-9])+INT(In[9]))
				              +   507*(INT(In[-8])+INT(In[8]))
				              +   776*(INT(In[-7])+INT(In[7]))
				              +   1122*(INT(In[-6])+INT(In[6]))
				              +   1532*(INT(In[-5])+INT(In[5]))
				              +   1978*(INT(In[-4])+INT(In[4]))
				              +   2412*(INT(In[-3])+INT(In[3]))
				              +   2779*(INT(In[-2])+INT(In[2]))
				              +   3026*(INT(In[-1])+INT(In[1]))
				              +   3112*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num40(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-17),-17,17,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num41 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num41(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-18])+INT(In[18]))
				              +   1*(INT(In[-17])+INT(In[17]))
				              +   3*(INT(In[-16])+INT(In[16]))
				              +   7*(INT(In[-15])+INT(In[15]))
				              +   15*(INT(In[-14])+INT(In[14]))
				              +   32*(INT(In[-13])+INT(In[13]))
				              +   62*(INT(In[-12])+INT(In[12]))
				              +   115*(INT(In[-11])+INT(In[11]))
				              +   204*(INT(In[-10])+INT(In[10]))
				              +   340*(INT(In[-9])+INT(In[9]))
				              +   539*(INT(In[-8])+INT(In[8]))
				              +   808*(INT(In[-7])+INT(In[7]))
				              +   1148*(INT(In[-6])+INT(In[6]))
				              +   1546*(INT(In[-5])+INT(In[5]))
				              +   1972*(INT(In[-4])+INT(In[4]))
				              +   2383*(INT(In[-3])+INT(In[3]))
				              +   2729*(INT(In[-2])+INT(In[2]))
				              +   2959*(INT(In[-1])+INT(In[1]))
				              +   3040*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num41(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-18),-18,18,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num42 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num42(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-18])+INT(In[18]))
				              +   2*(INT(In[-17])+INT(In[17]))
				              +   4*(INT(In[-16])+INT(In[16]))
				              +   9*(INT(In[-15])+INT(In[15]))
				              +   19*(INT(In[-14])+INT(In[14]))
				              +   38*(INT(In[-13])+INT(In[13]))
				              +   72*(INT(In[-12])+INT(In[12]))
				              +   131*(INT(In[-11])+INT(In[11]))
				              +   224*(INT(In[-10])+INT(In[10]))
				              +   367*(INT(In[-9])+INT(In[9]))
				              +   569*(INT(In[-8])+INT(In[8]))
				              +   838*(INT(In[-7])+INT(In[7]))
				              +   1173*(INT(In[-6])+INT(In[6]))
				              +   1558*(INT(In[-5])+INT(In[5]))
				              +   1965*(INT(In[-4])+INT(In[4]))
				              +   2355*(INT(In[-3])+INT(In[3]))
				              +   2679*(INT(In[-2])+INT(In[2]))
				              +   2895*(INT(In[-1])+INT(In[1]))
				              +   2970*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num42(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-18),-18,18,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num43 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num43(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-18])+INT(In[18]))
				              +   2*(INT(In[-17])+INT(In[17]))
				              +   5*(INT(In[-16])+INT(In[16]))
				              +   11*(INT(In[-15])+INT(In[15]))
				              +   23*(INT(In[-14])+INT(In[14]))
				              +   45*(INT(In[-13])+INT(In[13]))
				              +   83*(INT(In[-12])+INT(In[12]))
				              +   147*(INT(In[-11])+INT(In[11]))
				              +   246*(INT(In[-10])+INT(In[10]))
				              +   393*(INT(In[-9])+INT(In[9]))
				              +   598*(INT(In[-8])+INT(In[8]))
				              +   867*(INT(In[-7])+INT(In[7]))
				              +   1194*(INT(In[-6])+INT(In[6]))
				              +   1567*(INT(In[-5])+INT(In[5]))
				              +   1957*(INT(In[-4])+INT(In[4]))
				              +   2326*(INT(In[-3])+INT(In[3]))
				              +   2632*(INT(In[-2])+INT(In[2]))
				              +   2834*(INT(In[-1])+INT(In[1]))
				              +   2906*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num43(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-18),-18,18,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num44 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num44(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-19])+INT(In[19]))
				              +   1*(INT(In[-18])+INT(In[18]))
				              +   3*(INT(In[-17])+INT(In[17]))
				              +   7*(INT(In[-16])+INT(In[16]))
				              +   14*(INT(In[-15])+INT(In[15]))
				              +   28*(INT(In[-14])+INT(In[14]))
				              +   52*(INT(In[-13])+INT(In[13]))
				              +   95*(INT(In[-12])+INT(In[12]))
				              +   163*(INT(In[-11])+INT(In[11]))
				              +   267*(INT(In[-10])+INT(In[10]))
				              +   419*(INT(In[-9])+INT(In[9]))
				              +   626*(INT(In[-8])+INT(In[8]))
				              +   893*(INT(In[-7])+INT(In[7]))
				              +   1214*(INT(In[-6])+INT(In[6]))
				              +   1574*(INT(In[-5])+INT(In[5]))
				              +   1947*(INT(In[-4])+INT(In[4]))
				              +   2297*(INT(In[-3])+INT(In[3]))
				              +   2586*(INT(In[-2])+INT(In[2]))
				              +   2776*(INT(In[-1])+INT(In[1]))
				              +   2842*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num44(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-19),-19,19,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num45 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num45(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-19])+INT(In[19]))
				              +   2*(INT(In[-18])+INT(In[18]))
				              +   4*(INT(In[-17])+INT(In[17]))
				              +   8*(INT(In[-16])+INT(In[16]))
				              +   17*(INT(In[-15])+INT(In[15]))
				              +   33*(INT(In[-14])+INT(In[14]))
				              +   61*(INT(In[-13])+INT(In[13]))
				              +   107*(INT(In[-12])+INT(In[12]))
				              +   180*(INT(In[-11])+INT(In[11]))
				              +   289*(INT(In[-10])+INT(In[10]))
				              +   445*(INT(In[-9])+INT(In[9]))
				              +   653*(INT(In[-8])+INT(In[8]))
				              +   917*(INT(In[-7])+INT(In[7]))
				              +   1231*(INT(In[-6])+INT(In[6]))
				              +   1580*(INT(In[-5])+INT(In[5]))
				              +   1936*(INT(In[-4])+INT(In[4]))
				              +   2269*(INT(In[-3])+INT(In[3]))
				              +   2541*(INT(In[-2])+INT(In[2]))
				              +   2719*(INT(In[-1])+INT(In[1]))
				              +   2782*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num45(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-19),-19,19,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num46 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num46(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-19])+INT(In[19]))
				              +   2*(INT(In[-18])+INT(In[18]))
				              +   5*(INT(In[-17])+INT(In[17]))
				              +   11*(INT(In[-16])+INT(In[16]))
				              +   21*(INT(In[-15])+INT(In[15]))
				              +   39*(INT(In[-14])+INT(In[14]))
				              +   70*(INT(In[-13])+INT(In[13]))
				              +   120*(INT(In[-12])+INT(In[12]))
				              +   197*(INT(In[-11])+INT(In[11]))
				              +   311*(INT(In[-10])+INT(In[10]))
				              +   470*(INT(In[-9])+INT(In[9]))
				              +   679*(INT(In[-8])+INT(In[8]))
				              +   940*(INT(In[-7])+INT(In[7]))
				              +   1247*(INT(In[-6])+INT(In[6]))
				              +   1583*(INT(In[-5])+INT(In[5]))
				              +   1925*(INT(In[-4])+INT(In[4]))
				              +   2240*(INT(In[-3])+INT(In[3]))
				              +   2497*(INT(In[-2])+INT(In[2]))
				              +   2665*(INT(In[-1])+INT(In[1]))
				              +   2722*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num46(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-19),-19,19,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num47 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num47(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-20])+INT(In[20]))
				              +   1*(INT(In[-19])+INT(In[19]))
				              +   3*(INT(In[-18])+INT(In[18]))
				              +   6*(INT(In[-17])+INT(In[17]))
				              +   13*(INT(In[-16])+INT(In[16]))
				              +   24*(INT(In[-15])+INT(In[15]))
				              +   45*(INT(In[-14])+INT(In[14]))
				              +   79*(INT(In[-13])+INT(In[13]))
				              +   133*(INT(In[-12])+INT(In[12]))
				              +   215*(INT(In[-11])+INT(In[11]))
				              +   332*(INT(In[-10])+INT(In[10]))
				              +   494*(INT(In[-9])+INT(In[9]))
				              +   704*(INT(In[-8])+INT(In[8]))
				              +   962*(INT(In[-7])+INT(In[7]))
				              +   1261*(INT(In[-6])+INT(In[6]))
				              +   1585*(INT(In[-5])+INT(In[5]))
				              +   1912*(INT(In[-4])+INT(In[4]))
				              +   2212*(INT(In[-3])+INT(In[3]))
				              +   2455*(INT(In[-2])+INT(In[2]))
				              +   2613*(INT(In[-1])+INT(In[1]))
				              +   2668*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num47(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-20),-20,20,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num48 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num48(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-20])+INT(In[20]))
				              +   2*(INT(In[-19])+INT(In[19]))
				              +   4*(INT(In[-18])+INT(In[18]))
				              +   8*(INT(In[-17])+INT(In[17]))
				              +   16*(INT(In[-16])+INT(In[16]))
				              +   29*(INT(In[-15])+INT(In[15]))
				              +   52*(INT(In[-14])+INT(In[14]))
				              +   89*(INT(In[-13])+INT(In[13]))
				              +   147*(INT(In[-12])+INT(In[12]))
				              +   232*(INT(In[-11])+INT(In[11]))
				              +   354*(INT(In[-10])+INT(In[10]))
				              +   517*(INT(In[-9])+INT(In[9]))
				              +   727*(INT(In[-8])+INT(In[8]))
				              +   981*(INT(In[-7])+INT(In[7]))
				              +   1273*(INT(In[-6])+INT(In[6]))
				              +   1586*(INT(In[-5])+INT(In[5]))
				              +   1899*(INT(In[-4])+INT(In[4]))
				              +   2184*(INT(In[-3])+INT(In[3]))
				              +   2413*(INT(In[-2])+INT(In[2]))
				              +   2563*(INT(In[-1])+INT(In[1]))
				              +   2614*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num48(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-20),-20,20,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num49 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num49(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-21])+INT(In[21]))
				              +   1*(INT(In[-20])+INT(In[20]))
				              +   2*(INT(In[-19])+INT(In[19]))
				              +   5*(INT(In[-18])+INT(In[18]))
				              +   10*(INT(In[-17])+INT(In[17]))
				              +   19*(INT(In[-16])+INT(In[16]))
				              +   34*(INT(In[-15])+INT(In[15]))
				              +   59*(INT(In[-14])+INT(In[14]))
				              +   100*(INT(In[-13])+INT(In[13]))
				              +   161*(INT(In[-12])+INT(In[12]))
				              +   250*(INT(In[-11])+INT(In[11]))
				              +   375*(INT(In[-10])+INT(In[10]))
				              +   540*(INT(In[-9])+INT(In[9]))
				              +   749*(INT(In[-8])+INT(In[8]))
				              +   999*(INT(In[-7])+INT(In[7]))
				              +   1283*(INT(In[-6])+INT(In[6]))
				              +   1585*(INT(In[-5])+INT(In[5]))
				              +   1885*(INT(In[-4])+INT(In[4]))
				              +   2156*(INT(In[-3])+INT(In[3]))
				              +   2374*(INT(In[-2])+INT(In[2]))
				              +   2515*(INT(In[-1])+INT(In[1]))
				              +   2562*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num49(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-21),-21,21,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num50 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num50(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-21])+INT(In[21]))
				              +   2*(INT(In[-20])+INT(In[20]))
				              +   3*(INT(In[-19])+INT(In[19]))
				              +   6*(INT(In[-18])+INT(In[18]))
				              +   12*(INT(In[-17])+INT(In[17]))
				              +   22*(INT(In[-16])+INT(In[16]))
				              +   39*(INT(In[-15])+INT(In[15]))
				              +   67*(INT(In[-14])+INT(In[14]))
				              +   110*(INT(In[-13])+INT(In[13]))
				              +   175*(INT(In[-12])+INT(In[12]))
				              +   268*(INT(In[-11])+INT(In[11]))
				              +   396*(INT(In[-10])+INT(In[10]))
				              +   562*(INT(In[-9])+INT(In[9]))
				              +   770*(INT(In[-8])+INT(In[8]))
				              +   1016*(INT(In[-7])+INT(In[7]))
				              +   1292*(INT(In[-6])+INT(In[6]))
				              +   1584*(INT(In[-5])+INT(In[5]))
				              +   1870*(INT(In[-4])+INT(In[4]))
				              +   2129*(INT(In[-3])+INT(In[3]))
				              +   2335*(INT(In[-2])+INT(In[2]))
				              +   2468*(INT(In[-1])+INT(In[1]))
				              +   2514*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num50(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-21),-21,21,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num51 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num51(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-21])+INT(In[21]))
				              +   2*(INT(In[-20])+INT(In[20]))
				              +   4*(INT(In[-19])+INT(In[19]))
				              +   8*(INT(In[-18])+INT(In[18]))
				              +   14*(INT(In[-17])+INT(In[17]))
				              +   26*(INT(In[-16])+INT(In[16]))
				              +   45*(INT(In[-15])+INT(In[15]))
				              +   75*(INT(In[-14])+INT(In[14]))
				              +   122*(INT(In[-13])+INT(In[13]))
				              +   190*(INT(In[-12])+INT(In[12]))
				              +   286*(INT(In[-11])+INT(In[11]))
				              +   416*(INT(In[-10])+INT(In[10]))
				              +   583*(INT(In[-9])+INT(In[9]))
				              +   790*(INT(In[-8])+INT(In[8]))
				              +   1031*(INT(In[-7])+INT(In[7]))
				              +   1300*(INT(In[-6])+INT(In[6]))
				              +   1581*(INT(In[-5])+INT(In[5]))
				              +   1855*(INT(In[-4])+INT(In[4]))
				              +   2101*(INT(In[-3])+INT(In[3]))
				              +   2297*(INT(In[-2])+INT(In[2]))
				              +   2423*(INT(In[-1])+INT(In[1]))
				              +   2468*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num51(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-21),-21,21,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num52 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num52(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-22])+INT(In[22]))
				              +   1*(INT(In[-21])+INT(In[21]))
				              +   3*(INT(In[-20])+INT(In[20]))
				              +   5*(INT(In[-19])+INT(In[19]))
				              +   9*(INT(In[-18])+INT(In[18]))
				              +   17*(INT(In[-17])+INT(In[17]))
				              +   30*(INT(In[-16])+INT(In[16]))
				              +   51*(INT(In[-15])+INT(In[15]))
				              +   84*(INT(In[-14])+INT(In[14]))
				              +   133*(INT(In[-13])+INT(In[13]))
				              +   205*(INT(In[-12])+INT(In[12]))
				              +   304*(INT(In[-11])+INT(In[11]))
				              +   436*(INT(In[-10])+INT(In[10]))
				              +   604*(INT(In[-9])+INT(In[9]))
				              +   808*(INT(In[-8])+INT(In[8]))
				              +   1045*(INT(In[-7])+INT(In[7]))
				              +   1306*(INT(In[-6])+INT(In[6]))
				              +   1577*(INT(In[-5])+INT(In[5]))
				              +   1840*(INT(In[-4])+INT(In[4]))
				              +   2075*(INT(In[-3])+INT(In[3]))
				              +   2260*(INT(In[-2])+INT(In[2]))
				              +   2380*(INT(In[-1])+INT(In[1]))
				              +   2420*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num52(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-22),-22,22,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num53 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num53(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-22])+INT(In[22]))
				              +   2*(INT(In[-21])+INT(In[21]))
				              +   3*(INT(In[-20])+INT(In[20]))
				              +   6*(INT(In[-19])+INT(In[19]))
				              +   11*(INT(In[-18])+INT(In[18]))
				              +   20*(INT(In[-17])+INT(In[17]))
				              +   35*(INT(In[-16])+INT(In[16]))
				              +   58*(INT(In[-15])+INT(In[15]))
				              +   93*(INT(In[-14])+INT(In[14]))
				              +   145*(INT(In[-13])+INT(In[13]))
				              +   220*(INT(In[-12])+INT(In[12]))
				              +   322*(INT(In[-11])+INT(In[11]))
				              +   455*(INT(In[-10])+INT(In[10]))
				              +   623*(INT(In[-9])+INT(In[9]))
				              +   825*(INT(In[-8])+INT(In[8]))
				              +   1057*(INT(In[-7])+INT(In[7]))
				              +   1311*(INT(In[-6])+INT(In[6]))
				              +   1572*(INT(In[-5])+INT(In[5]))
				              +   1825*(INT(In[-4])+INT(In[4]))
				              +   2048*(INT(In[-3])+INT(In[3]))
				              +   2225*(INT(In[-2])+INT(In[2]))
				              +   2338*(INT(In[-1])+INT(In[1]))
				              +   2378*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num53(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-22),-22,22,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num54 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num54(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-23])+INT(In[23]))
				              +   1*(INT(In[-22])+INT(In[22]))
				              +   2*(INT(In[-21])+INT(In[21]))
				              +   4*(INT(In[-20])+INT(In[20]))
				              +   7*(INT(In[-19])+INT(In[19]))
				              +   13*(INT(In[-18])+INT(In[18]))
				              +   23*(INT(In[-17])+INT(In[17]))
				              +   39*(INT(In[-16])+INT(In[16]))
				              +   65*(INT(In[-15])+INT(In[15]))
				              +   103*(INT(In[-14])+INT(In[14]))
				              +   158*(INT(In[-13])+INT(In[13]))
				              +   235*(INT(In[-12])+INT(In[12]))
				              +   339*(INT(In[-11])+INT(In[11]))
				              +   474*(INT(In[-10])+INT(In[10]))
				              +   642*(INT(In[-9])+INT(In[9]))
				              +   841*(INT(In[-8])+INT(In[8]))
				              +   1069*(INT(In[-7])+INT(In[7]))
				              +   1315*(INT(In[-6])+INT(In[6]))
				              +   1567*(INT(In[-5])+INT(In[5]))
				              +   1809*(INT(In[-4])+INT(In[4]))
				              +   2022*(INT(In[-3])+INT(In[3]))
				              +   2190*(INT(In[-2])+INT(In[2]))
				              +   2298*(INT(In[-1])+INT(In[1]))
				              +   2334*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num54(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-23),-23,23,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num55 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num55(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-23])+INT(In[23]))
				              +   1*(INT(In[-22])+INT(In[22]))
				              +   3*(INT(In[-21])+INT(In[21]))
				              +   5*(INT(In[-20])+INT(In[20]))
				              +   9*(INT(In[-19])+INT(In[19]))
				              +   16*(INT(In[-18])+INT(In[18]))
				              +   27*(INT(In[-17])+INT(In[17]))
				              +   45*(INT(In[-16])+INT(In[16]))
				              +   72*(INT(In[-15])+INT(In[15]))
				              +   112*(INT(In[-14])+INT(In[14]))
				              +   170*(INT(In[-13])+INT(In[13]))
				              +   250*(INT(In[-12])+INT(In[12]))
				              +   356*(INT(In[-11])+INT(In[11]))
				              +   492*(INT(In[-10])+INT(In[10]))
				              +   659*(INT(In[-9])+INT(In[9]))
				              +   857*(INT(In[-8])+INT(In[8]))
				              +   1079*(INT(In[-7])+INT(In[7]))
				              +   1318*(INT(In[-6])+INT(In[6]))
				              +   1561*(INT(In[-5])+INT(In[5]))
				              +   1793*(INT(In[-4])+INT(In[4]))
				              +   1997*(INT(In[-3])+INT(In[3]))
				              +   2157*(INT(In[-2])+INT(In[2]))
				              +   2258*(INT(In[-1])+INT(In[1]))
				              +   2292*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num55(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-23),-23,23,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num56 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num56(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-23])+INT(In[23]))
				              +   2*(INT(In[-22])+INT(In[22]))
				              +   3*(INT(In[-21])+INT(In[21]))
				              +   6*(INT(In[-20])+INT(In[20]))
				              +   11*(INT(In[-19])+INT(In[19]))
				              +   18*(INT(In[-18])+INT(In[18]))
				              +   31*(INT(In[-17])+INT(In[17]))
				              +   50*(INT(In[-16])+INT(In[16]))
				              +   79*(INT(In[-15])+INT(In[15]))
				              +   122*(INT(In[-14])+INT(In[14]))
				              +   183*(INT(In[-13])+INT(In[13]))
				              +   265*(INT(In[-12])+INT(In[12]))
				              +   373*(INT(In[-11])+INT(In[11]))
				              +   510*(INT(In[-10])+INT(In[10]))
				              +   676*(INT(In[-9])+INT(In[9]))
				              +   871*(INT(In[-8])+INT(In[8]))
				              +   1088*(INT(In[-7])+INT(In[7]))
				              +   1320*(INT(In[-6])+INT(In[6]))
				              +   1554*(INT(In[-5])+INT(In[5]))
				              +   1777*(INT(In[-4])+INT(In[4]))
				              +   1972*(INT(In[-3])+INT(In[3]))
				              +   2124*(INT(In[-2])+INT(In[2]))
				              +   2221*(INT(In[-1])+INT(In[1]))
				              +   2254*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num56(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-23),-23,23,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num57 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num57(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-24])+INT(In[24]))
				              +   1*(INT(In[-23])+INT(In[23]))
				              +   2*(INT(In[-22])+INT(In[22]))
				              +   4*(INT(In[-21])+INT(In[21]))
				              +   7*(INT(In[-20])+INT(In[20]))
				              +   12*(INT(In[-19])+INT(In[19]))
				              +   21*(INT(In[-18])+INT(In[18]))
				              +   35*(INT(In[-17])+INT(In[17]))
				              +   56*(INT(In[-16])+INT(In[16]))
				              +   87*(INT(In[-15])+INT(In[15]))
				              +   133*(INT(In[-14])+INT(In[14]))
				              +   196*(INT(In[-13])+INT(In[13]))
				              +   280*(INT(In[-12])+INT(In[12]))
				              +   390*(INT(In[-11])+INT(In[11]))
				              +   527*(INT(In[-10])+INT(In[10]))
				              +   692*(INT(In[-9])+INT(In[9]))
				              +   884*(INT(In[-8])+INT(In[8]))
				              +   1096*(INT(In[-7])+INT(In[7]))
				              +   1321*(INT(In[-6])+INT(In[6]))
				              +   1547*(INT(In[-5])+INT(In[5]))
				              +   1761*(INT(In[-4])+INT(In[4]))
				              +   1947*(INT(In[-3])+INT(In[3]))
				              +   2092*(INT(In[-2])+INT(In[2]))
				              +   2184*(INT(In[-1])+INT(In[1]))
				              +   2216*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num57(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-24),-24,24,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num58 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num58(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-24])+INT(In[24]))
				              +   1*(INT(In[-23])+INT(In[23]))
				              +   3*(INT(In[-22])+INT(In[22]))
				              +   5*(INT(In[-21])+INT(In[21]))
				              +   8*(INT(In[-20])+INT(In[20]))
				              +   14*(INT(In[-19])+INT(In[19]))
				              +   24*(INT(In[-18])+INT(In[18]))
				              +   39*(INT(In[-17])+INT(In[17]))
				              +   62*(INT(In[-16])+INT(In[16]))
				              +   96*(INT(In[-15])+INT(In[15]))
				              +   143*(INT(In[-14])+INT(In[14]))
				              +   208*(INT(In[-13])+INT(In[13]))
				              +   295*(INT(In[-12])+INT(In[12]))
				              +   406*(INT(In[-11])+INT(In[11]))
				              +   543*(INT(In[-10])+INT(In[10]))
				              +   707*(INT(In[-9])+INT(In[9]))
				              +   896*(INT(In[-8])+INT(In[8]))
				              +   1103*(INT(In[-7])+INT(In[7]))
				              +   1322*(INT(In[-6])+INT(In[6]))
				              +   1540*(INT(In[-5])+INT(In[5]))
				              +   1745*(INT(In[-4])+INT(In[4]))
				              +   1923*(INT(In[-3])+INT(In[3]))
				              +   2061*(INT(In[-2])+INT(In[2]))
				              +   2149*(INT(In[-1])+INT(In[1]))
				              +   2180*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num58(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-24),-24,24,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num59 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num59(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-24])+INT(In[24]))
				              +   2*(INT(In[-23])+INT(In[23]))
				              +   3*(INT(In[-22])+INT(In[22]))
				              +   6*(INT(In[-21])+INT(In[21]))
				              +   10*(INT(In[-20])+INT(In[20]))
				              +   17*(INT(In[-19])+INT(In[19]))
				              +   28*(INT(In[-18])+INT(In[18]))
				              +   44*(INT(In[-17])+INT(In[17]))
				              +   69*(INT(In[-16])+INT(In[16]))
				              +   104*(INT(In[-15])+INT(In[15]))
				              +   154*(INT(In[-14])+INT(In[14]))
				              +   221*(INT(In[-13])+INT(In[13]))
				              +   310*(INT(In[-12])+INT(In[12]))
				              +   422*(INT(In[-11])+INT(In[11]))
				              +   559*(INT(In[-10])+INT(In[10]))
				              +   722*(INT(In[-9])+INT(In[9]))
				              +   907*(INT(In[-8])+INT(In[8]))
				              +   1109*(INT(In[-7])+INT(In[7]))
				              +   1321*(INT(In[-6])+INT(In[6]))
				              +   1532*(INT(In[-5])+INT(In[5]))
				              +   1728*(INT(In[-4])+INT(In[4]))
				              +   1899*(INT(In[-3])+INT(In[3]))
				              +   2031*(INT(In[-2])+INT(In[2]))
				              +   2114*(INT(In[-1])+INT(In[1]))
				              +   2142*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num59(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-24),-24,24,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num60 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num60(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-25])+INT(In[25]))
				              +   1*(INT(In[-24])+INT(In[24]))
				              +   2*(INT(In[-23])+INT(In[23]))
				              +   4*(INT(In[-22])+INT(In[22]))
				              +   7*(INT(In[-21])+INT(In[21]))
				              +   12*(INT(In[-20])+INT(In[20]))
				              +   19*(INT(In[-19])+INT(In[19]))
				              +   31*(INT(In[-18])+INT(In[18]))
				              +   49*(INT(In[-17])+INT(In[17]))
				              +   75*(INT(In[-16])+INT(In[16]))
				              +   113*(INT(In[-15])+INT(In[15]))
				              +   165*(INT(In[-14])+INT(In[14]))
				              +   234*(INT(In[-13])+INT(In[13]))
				              +   324*(INT(In[-12])+INT(In[12]))
				              +   437*(INT(In[-11])+INT(In[11]))
				              +   574*(INT(In[-10])+INT(In[10]))
				              +   735*(INT(In[-9])+INT(In[9]))
				              +   917*(INT(In[-8])+INT(In[8]))
				              +   1115*(INT(In[-7])+INT(In[7]))
				              +   1320*(INT(In[-6])+INT(In[6]))
				              +   1523*(INT(In[-5])+INT(In[5]))
				              +   1712*(INT(In[-4])+INT(In[4]))
				              +   1876*(INT(In[-3])+INT(In[3]))
				              +   2002*(INT(In[-2])+INT(In[2]))
				              +   2081*(INT(In[-1])+INT(In[1]))
				              +   2110*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num60(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-25),-25,25,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num61 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num61(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-25])+INT(In[25]))
				              +   1*(INT(In[-24])+INT(In[24]))
				              +   3*(INT(In[-23])+INT(In[23]))
				              +   5*(INT(In[-22])+INT(In[22]))
				              +   8*(INT(In[-21])+INT(In[21]))
				              +   13*(INT(In[-20])+INT(In[20]))
				              +   22*(INT(In[-19])+INT(In[19]))
				              +   35*(INT(In[-18])+INT(In[18]))
				              +   54*(INT(In[-17])+INT(In[17]))
				              +   83*(INT(In[-16])+INT(In[16]))
				              +   122*(INT(In[-15])+INT(In[15]))
				              +   176*(INT(In[-14])+INT(In[14]))
				              +   247*(INT(In[-13])+INT(In[13]))
				              +   338*(INT(In[-12])+INT(In[12]))
				              +   452*(INT(In[-11])+INT(In[11]))
				              +   589*(INT(In[-10])+INT(In[10]))
				              +   748*(INT(In[-9])+INT(In[9]))
				              +   927*(INT(In[-8])+INT(In[8]))
				              +   1119*(INT(In[-7])+INT(In[7]))
				              +   1318*(INT(In[-6])+INT(In[6]))
				              +   1514*(INT(In[-5])+INT(In[5]))
				              +   1696*(INT(In[-4])+INT(In[4]))
				              +   1853*(INT(In[-3])+INT(In[3]))
				              +   1973*(INT(In[-2])+INT(In[2]))
				              +   2049*(INT(In[-1])+INT(In[1]))
				              +   2076*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num61(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-25),-25,25,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num62 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num62(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-26])+INT(In[26]))
				              +   1*(INT(In[-25])+INT(In[25]))
				              +   2*(INT(In[-24])+INT(In[24]))
				              +   3*(INT(In[-23])+INT(In[23]))
				              +   6*(INT(In[-22])+INT(In[22]))
				              +   9*(INT(In[-21])+INT(In[21]))
				              +   15*(INT(In[-20])+INT(In[20]))
				              +   25*(INT(In[-19])+INT(In[19]))
				              +   39*(INT(In[-18])+INT(In[18]))
				              +   60*(INT(In[-17])+INT(In[17]))
				              +   90*(INT(In[-16])+INT(In[16]))
				              +   131*(INT(In[-15])+INT(In[15]))
				              +   187*(INT(In[-14])+INT(In[14]))
				              +   260*(INT(In[-13])+INT(In[13]))
				              +   352*(INT(In[-12])+INT(In[12]))
				              +   466*(INT(In[-11])+INT(In[11]))
				              +   603*(INT(In[-10])+INT(In[10]))
				              +   760*(INT(In[-9])+INT(In[9]))
				              +   935*(INT(In[-8])+INT(In[8]))
				              +   1123*(INT(In[-7])+INT(In[7]))
				              +   1316*(INT(In[-6])+INT(In[6]))
				              +   1505*(INT(In[-5])+INT(In[5]))
				              +   1680*(INT(In[-4])+INT(In[4]))
				              +   1830*(INT(In[-3])+INT(In[3]))
				              +   1945*(INT(In[-2])+INT(In[2]))
				              +   2018*(INT(In[-1])+INT(In[1]))
				              +   2044*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num62(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-26),-26,26,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num63 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num63(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-26])+INT(In[26]))
				              +   1*(INT(In[-25])+INT(In[25]))
				              +   2*(INT(In[-24])+INT(In[24]))
				              +   4*(INT(In[-23])+INT(In[23]))
				              +   7*(INT(In[-22])+INT(In[22]))
				              +   11*(INT(In[-21])+INT(In[21]))
				              +   18*(INT(In[-20])+INT(In[20]))
				              +   28*(INT(In[-19])+INT(In[19]))
				              +   43*(INT(In[-18])+INT(In[18]))
				              +   66*(INT(In[-17])+INT(In[17]))
				              +   97*(INT(In[-16])+INT(In[16]))
				              +   140*(INT(In[-15])+INT(In[15]))
				              +   198*(INT(In[-14])+INT(In[14]))
				              +   272*(INT(In[-13])+INT(In[13]))
				              +   366*(INT(In[-12])+INT(In[12]))
				              +   480*(INT(In[-11])+INT(In[11]))
				              +   616*(INT(In[-10])+INT(In[10]))
				              +   771*(INT(In[-9])+INT(In[9]))
				              +   943*(INT(In[-8])+INT(In[8]))
				              +   1126*(INT(In[-7])+INT(In[7]))
				              +   1314*(INT(In[-6])+INT(In[6]))
				              +   1496*(INT(In[-5])+INT(In[5]))
				              +   1664*(INT(In[-4])+INT(In[4]))
				              +   1808*(INT(In[-3])+INT(In[3]))
				              +   1918*(INT(In[-2])+INT(In[2]))
				              +   1988*(INT(In[-1])+INT(In[1]))
				              +   2012*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num63(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-26),-26,26,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num64 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num64(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-26])+INT(In[26]))
				              +   1*(INT(In[-25])+INT(In[25]))
				              +   3*(INT(In[-24])+INT(In[24]))
				              +   5*(INT(In[-23])+INT(In[23]))
				              +   8*(INT(In[-22])+INT(In[22]))
				              +   13*(INT(In[-21])+INT(In[21]))
				              +   20*(INT(In[-20])+INT(In[20]))
				              +   31*(INT(In[-19])+INT(In[19]))
				              +   48*(INT(In[-18])+INT(In[18]))
				              +   72*(INT(In[-17])+INT(In[17]))
				              +   105*(INT(In[-16])+INT(In[16]))
				              +   150*(INT(In[-15])+INT(In[15]))
				              +   209*(INT(In[-14])+INT(In[14]))
				              +   285*(INT(In[-13])+INT(In[13]))
				              +   379*(INT(In[-12])+INT(In[12]))
				              +   494*(INT(In[-11])+INT(In[11]))
				              +   628*(INT(In[-10])+INT(In[10]))
				              +   782*(INT(In[-9])+INT(In[9]))
				              +   950*(INT(In[-8])+INT(In[8]))
				              +   1129*(INT(In[-7])+INT(In[7]))
				              +   1310*(INT(In[-6])+INT(In[6]))
				              +   1487*(INT(In[-5])+INT(In[5]))
				              +   1648*(INT(In[-4])+INT(In[4]))
				              +   1786*(INT(In[-3])+INT(In[3]))
				              +   1892*(INT(In[-2])+INT(In[2]))
				              +   1958*(INT(In[-1])+INT(In[1]))
				              +   1980*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num64(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-26),-26,26,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num65 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num65(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-27])+INT(In[27]))
				              +   1*(INT(In[-26])+INT(In[26]))
				              +   2*(INT(In[-25])+INT(In[25]))
				              +   3*(INT(In[-24])+INT(In[24]))
				              +   5*(INT(In[-23])+INT(In[23]))
				              +   9*(INT(In[-22])+INT(In[22]))
				              +   14*(INT(In[-21])+INT(In[21]))
				              +   23*(INT(In[-20])+INT(In[20]))
				              +   35*(INT(In[-19])+INT(In[19]))
				              +   53*(INT(In[-18])+INT(In[18]))
				              +   78*(INT(In[-17])+INT(In[17]))
				              +   113*(INT(In[-16])+INT(In[16]))
				              +   159*(INT(In[-15])+INT(In[15]))
				              +   220*(INT(In[-14])+INT(In[14]))
				              +   297*(INT(In[-13])+INT(In[13]))
				              +   392*(INT(In[-12])+INT(In[12]))
				              +   507*(INT(In[-11])+INT(In[11]))
				              +   641*(INT(In[-10])+INT(In[10]))
				              +   791*(INT(In[-9])+INT(In[9]))
				              +   957*(INT(In[-8])+INT(In[8]))
				              +   1130*(INT(In[-7])+INT(In[7]))
				              +   1307*(INT(In[-6])+INT(In[6]))
				              +   1477*(INT(In[-5])+INT(In[5]))
				              +   1633*(INT(In[-4])+INT(In[4]))
				              +   1765*(INT(In[-3])+INT(In[3]))
				              +   1866*(INT(In[-2])+INT(In[2]))
				              +   1930*(INT(In[-1])+INT(In[1]))
				              +   1950*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num65(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-27),-27,27,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num66 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num66(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-27])+INT(In[27]))
				              +   1*(INT(In[-26])+INT(In[26]))
				              +   2*(INT(In[-25])+INT(In[25]))
				              +   4*(INT(In[-24])+INT(In[24]))
				              +   6*(INT(In[-23])+INT(In[23]))
				              +   10*(INT(In[-22])+INT(In[22]))
				              +   16*(INT(In[-21])+INT(In[21]))
				              +   25*(INT(In[-20])+INT(In[20]))
				              +   39*(INT(In[-19])+INT(In[19]))
				              +   58*(INT(In[-18])+INT(In[18]))
				              +   84*(INT(In[-17])+INT(In[17]))
				              +   121*(INT(In[-16])+INT(In[16]))
				              +   169*(INT(In[-15])+INT(In[15]))
				              +   231*(INT(In[-14])+INT(In[14]))
				              +   309*(INT(In[-13])+INT(In[13]))
				              +   405*(INT(In[-12])+INT(In[12]))
				              +   520*(INT(In[-11])+INT(In[11]))
				              +   652*(INT(In[-10])+INT(In[10]))
				              +   801*(INT(In[-9])+INT(In[9]))
				              +   962*(INT(In[-8])+INT(In[8]))
				              +   1132*(INT(In[-7])+INT(In[7]))
				              +   1303*(INT(In[-6])+INT(In[6]))
				              +   1467*(INT(In[-5])+INT(In[5]))
				              +   1617*(INT(In[-4])+INT(In[4]))
				              +   1744*(INT(In[-3])+INT(In[3]))
				              +   1841*(INT(In[-2])+INT(In[2]))
				              +   1902*(INT(In[-1])+INT(In[1]))
				              +   1924*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num66(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-27),-27,27,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num67 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num67(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   1*(INT(In[-27])+INT(In[27]))
				              +   2*(INT(In[-26])+INT(In[26]))
				              +   3*(INT(In[-25])+INT(In[25]))
				              +   4*(INT(In[-24])+INT(In[24]))
				              +   7*(INT(In[-23])+INT(In[23]))
				              +   12*(INT(In[-22])+INT(In[22]))
				              +   18*(INT(In[-21])+INT(In[21]))
				              +   28*(INT(In[-20])+INT(In[20]))
				              +   43*(INT(In[-19])+INT(In[19]))
				              +   63*(INT(In[-18])+INT(In[18]))
				              +   91*(INT(In[-17])+INT(In[17]))
				              +   129*(INT(In[-16])+INT(In[16]))
				              +   178*(INT(In[-15])+INT(In[15]))
				              +   242*(INT(In[-14])+INT(In[14]))
				              +   321*(INT(In[-13])+INT(In[13]))
				              +   418*(INT(In[-12])+INT(In[12]))
				              +   532*(INT(In[-11])+INT(In[11]))
				              +   663*(INT(In[-10])+INT(In[10]))
				              +   809*(INT(In[-9])+INT(In[9]))
				              +   967*(INT(In[-8])+INT(In[8]))
				              +   1132*(INT(In[-7])+INT(In[7]))
				              +   1298*(INT(In[-6])+INT(In[6]))
				              +   1457*(INT(In[-5])+INT(In[5]))
				              +   1602*(INT(In[-4])+INT(In[4]))
				              +   1724*(INT(In[-3])+INT(In[3]))
				              +   1817*(INT(In[-2])+INT(In[2]))
				              +   1875*(INT(In[-1])+INT(In[1]))
				              +   1894*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num67(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-28),-28,28,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num68 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num68(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   1*(INT(In[-27])+INT(In[27]))
				              +   2*(INT(In[-26])+INT(In[26]))
				              +   3*(INT(In[-25])+INT(In[25]))
				              +   5*(INT(In[-24])+INT(In[24]))
				              +   8*(INT(In[-23])+INT(In[23]))
				              +   13*(INT(In[-22])+INT(In[22]))
				              +   21*(INT(In[-21])+INT(In[21]))
				              +   32*(INT(In[-20])+INT(In[20]))
				              +   47*(INT(In[-19])+INT(In[19]))
				              +   68*(INT(In[-18])+INT(In[18]))
				              +   98*(INT(In[-17])+INT(In[17]))
				              +   137*(INT(In[-16])+INT(In[16]))
				              +   188*(INT(In[-15])+INT(In[15]))
				              +   253*(INT(In[-14])+INT(In[14]))
				              +   333*(INT(In[-13])+INT(In[13]))
				              +   430*(INT(In[-12])+INT(In[12]))
				              +   543*(INT(In[-11])+INT(In[11]))
				              +   673*(INT(In[-10])+INT(In[10]))
				              +   817*(INT(In[-9])+INT(In[9]))
				              +   972*(INT(In[-8])+INT(In[8]))
				              +   1133*(INT(In[-7])+INT(In[7]))
				              +   1293*(INT(In[-6])+INT(In[6]))
				              +   1447*(INT(In[-5])+INT(In[5]))
				              +   1586*(INT(In[-4])+INT(In[4]))
				              +   1704*(INT(In[-3])+INT(In[3]))
				              +   1793*(INT(In[-2])+INT(In[2]))
				              +   1849*(INT(In[-1])+INT(In[1]))
				              +   1868*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num68(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-28),-28,28,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num69 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num69(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   1*(INT(In[-27])+INT(In[27]))
				              +   2*(INT(In[-26])+INT(In[26]))
				              +   4*(INT(In[-25])+INT(In[25]))
				              +   6*(INT(In[-24])+INT(In[24]))
				              +   10*(INT(In[-23])+INT(In[23]))
				              +   15*(INT(In[-22])+INT(In[22]))
				              +   23*(INT(In[-21])+INT(In[21]))
				              +   35*(INT(In[-20])+INT(In[20]))
				              +   51*(INT(In[-19])+INT(In[19]))
				              +   74*(INT(In[-18])+INT(In[18]))
				              +   105*(INT(In[-17])+INT(In[17]))
				              +   145*(INT(In[-16])+INT(In[16]))
				              +   198*(INT(In[-15])+INT(In[15]))
				              +   264*(INT(In[-14])+INT(In[14]))
				              +   344*(INT(In[-13])+INT(In[13]))
				              +   441*(INT(In[-12])+INT(In[12]))
				              +   554*(INT(In[-11])+INT(In[11]))
				              +   683*(INT(In[-10])+INT(In[10]))
				              +   825*(INT(In[-9])+INT(In[9]))
				              +   976*(INT(In[-8])+INT(In[8]))
				              +   1133*(INT(In[-7])+INT(In[7]))
				              +   1288*(INT(In[-6])+INT(In[6]))
				              +   1437*(INT(In[-5])+INT(In[5]))
				              +   1571*(INT(In[-4])+INT(In[4]))
				              +   1684*(INT(In[-3])+INT(In[3]))
				              +   1770*(INT(In[-2])+INT(In[2]))
				              +   1823*(INT(In[-1])+INT(In[1]))
				              +   1842*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num69(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-28),-28,28,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num70 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num70(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   2*(INT(In[-27])+INT(In[27]))
				              +   3*(INT(In[-26])+INT(In[26]))
				              +   4*(INT(In[-25])+INT(In[25]))
				              +   7*(INT(In[-24])+INT(In[24]))
				              +   11*(INT(In[-23])+INT(In[23]))
				              +   17*(INT(In[-22])+INT(In[22]))
				              +   26*(INT(In[-21])+INT(In[21]))
				              +   38*(INT(In[-20])+INT(In[20]))
				              +   56*(INT(In[-19])+INT(In[19]))
				              +   80*(INT(In[-18])+INT(In[18]))
				              +   112*(INT(In[-17])+INT(In[17]))
				              +   154*(INT(In[-16])+INT(In[16]))
				              +   207*(INT(In[-15])+INT(In[15]))
				              +   274*(INT(In[-14])+INT(In[14]))
				              +   356*(INT(In[-13])+INT(In[13]))
				              +   453*(INT(In[-12])+INT(In[12]))
				              +   565*(INT(In[-11])+INT(In[11]))
				              +   692*(INT(In[-10])+INT(In[10]))
				              +   831*(INT(In[-9])+INT(In[9]))
				              +   979*(INT(In[-8])+INT(In[8]))
				              +   1132*(INT(In[-7])+INT(In[7]))
				              +   1283*(INT(In[-6])+INT(In[6]))
				              +   1427*(INT(In[-5])+INT(In[5]))
				              +   1556*(INT(In[-4])+INT(In[4]))
				              +   1665*(INT(In[-3])+INT(In[3]))
				              +   1747*(INT(In[-2])+INT(In[2]))
				              +   1798*(INT(In[-1])+INT(In[1]))
				              +   1816*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num70(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-28),-28,28,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num71 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num71(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-29])+INT(In[29]))
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   2*(INT(In[-27])+INT(In[27]))
				              +   3*(INT(In[-26])+INT(In[26]))
				              +   5*(INT(In[-25])+INT(In[25]))
				              +   8*(INT(In[-24])+INT(In[24]))
				              +   13*(INT(In[-23])+INT(In[23]))
				              +   19*(INT(In[-22])+INT(In[22]))
				              +   29*(INT(In[-21])+INT(In[21]))
				              +   42*(INT(In[-20])+INT(In[20]))
				              +   61*(INT(In[-19])+INT(In[19]))
				              +   86*(INT(In[-18])+INT(In[18]))
				              +   119*(INT(In[-17])+INT(In[17]))
				              +   162*(INT(In[-16])+INT(In[16]))
				              +   217*(INT(In[-15])+INT(In[15]))
				              +   285*(INT(In[-14])+INT(In[14]))
				              +   367*(INT(In[-13])+INT(In[13]))
				              +   464*(INT(In[-12])+INT(In[12]))
				              +   575*(INT(In[-11])+INT(In[11]))
				              +   701*(INT(In[-10])+INT(In[10]))
				              +   837*(INT(In[-9])+INT(In[9]))
				              +   982*(INT(In[-8])+INT(In[8]))
				              +   1131*(INT(In[-7])+INT(In[7]))
				              +   1277*(INT(In[-6])+INT(In[6]))
				              +   1416*(INT(In[-5])+INT(In[5]))
				              +   1541*(INT(In[-4])+INT(In[4]))
				              +   1646*(INT(In[-3])+INT(In[3]))
				              +   1725*(INT(In[-2])+INT(In[2]))
				              +   1774*(INT(In[-1])+INT(In[1]))
				              +   1790*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num71(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-29),-29,29,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num72 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num72(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-29])+INT(In[29]))
				              +   1*(INT(In[-28])+INT(In[28]))
				              +   2*(INT(In[-27])+INT(In[27]))
				              +   4*(INT(In[-26])+INT(In[26]))
				              +   6*(INT(In[-25])+INT(In[25]))
				              +   9*(INT(In[-24])+INT(In[24]))
				              +   14*(INT(In[-23])+INT(In[23]))
				              +   21*(INT(In[-22])+INT(In[22]))
				              +   32*(INT(In[-21])+INT(In[21]))
				              +   46*(INT(In[-20])+INT(In[20]))
				              +   65*(INT(In[-19])+INT(In[19]))
				              +   92*(INT(In[-18])+INT(In[18]))
				              +   126*(INT(In[-17])+INT(In[17]))
				              +   171*(INT(In[-16])+INT(In[16]))
				              +   226*(INT(In[-15])+INT(In[15]))
				              +   295*(INT(In[-14])+INT(In[14]))
				              +   378*(INT(In[-13])+INT(In[13]))
				              +   474*(INT(In[-12])+INT(In[12]))
				              +   585*(INT(In[-11])+INT(In[11]))
				              +   709*(INT(In[-10])+INT(In[10]))
				              +   843*(INT(In[-9])+INT(In[9]))
				              +   985*(INT(In[-8])+INT(In[8]))
				              +   1129*(INT(In[-7])+INT(In[7]))
				              +   1272*(INT(In[-6])+INT(In[6]))
				              +   1406*(INT(In[-5])+INT(In[5]))
				              +   1527*(INT(In[-4])+INT(In[4]))
				              +   1627*(INT(In[-3])+INT(In[3]))
				              +   1703*(INT(In[-2])+INT(In[2]))
				              +   1751*(INT(In[-1])+INT(In[1]))
				              +   1768*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num72(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-29),-29,29,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num73 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num73(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-29])+INT(In[29]))
				              +   2*(INT(In[-28])+INT(In[28]))
				              +   3*(INT(In[-27])+INT(In[27]))
				              +   4*(INT(In[-26])+INT(In[26]))
				              +   7*(INT(In[-25])+INT(In[25]))
				              +   10*(INT(In[-24])+INT(In[24]))
				              +   16*(INT(In[-23])+INT(In[23]))
				              +   24*(INT(In[-22])+INT(In[22]))
				              +   35*(INT(In[-21])+INT(In[21]))
				              +   50*(INT(In[-20])+INT(In[20]))
				              +   70*(INT(In[-19])+INT(In[19]))
				              +   98*(INT(In[-18])+INT(In[18]))
				              +   133*(INT(In[-17])+INT(In[17]))
				              +   179*(INT(In[-16])+INT(In[16]))
				              +   236*(INT(In[-15])+INT(In[15]))
				              +   305*(INT(In[-14])+INT(In[14]))
				              +   388*(INT(In[-13])+INT(In[13]))
				              +   485*(INT(In[-12])+INT(In[12]))
				              +   594*(INT(In[-11])+INT(In[11]))
				              +   717*(INT(In[-10])+INT(In[10]))
				              +   848*(INT(In[-9])+INT(In[9]))
				              +   987*(INT(In[-8])+INT(In[8]))
				              +   1128*(INT(In[-7])+INT(In[7]))
				              +   1266*(INT(In[-6])+INT(In[6]))
				              +   1396*(INT(In[-5])+INT(In[5]))
				              +   1512*(INT(In[-4])+INT(In[4]))
				              +   1609*(INT(In[-3])+INT(In[3]))
				              +   1682*(INT(In[-2])+INT(In[2]))
				              +   1728*(INT(In[-1])+INT(In[1]))
				              +   1742*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num73(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-29),-29,29,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num74 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num74(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-30])+INT(In[30]))
				              +   1*(INT(In[-29])+INT(In[29]))
				              +   2*(INT(In[-28])+INT(In[28]))
				              +   3*(INT(In[-27])+INT(In[27]))
				              +   5*(INT(In[-26])+INT(In[26]))
				              +   8*(INT(In[-25])+INT(In[25]))
				              +   12*(INT(In[-24])+INT(In[24]))
				              +   18*(INT(In[-23])+INT(In[23]))
				              +   26*(INT(In[-22])+INT(In[22]))
				              +   38*(INT(In[-21])+INT(In[21]))
				              +   54*(INT(In[-20])+INT(In[20]))
				              +   76*(INT(In[-19])+INT(In[19]))
				              +   104*(INT(In[-18])+INT(In[18]))
				              +   141*(INT(In[-17])+INT(In[17]))
				              +   188*(INT(In[-16])+INT(In[16]))
				              +   245*(INT(In[-15])+INT(In[15]))
				              +   315*(INT(In[-14])+INT(In[14]))
				              +   398*(INT(In[-13])+INT(In[13]))
				              +   495*(INT(In[-12])+INT(In[12]))
				              +   603*(INT(In[-11])+INT(In[11]))
				              +   724*(INT(In[-10])+INT(In[10]))
				              +   853*(INT(In[-9])+INT(In[9]))
				              +   988*(INT(In[-8])+INT(In[8]))
				              +   1125*(INT(In[-7])+INT(In[7]))
				              +   1260*(INT(In[-6])+INT(In[6]))
				              +   1385*(INT(In[-5])+INT(In[5]))
				              +   1498*(INT(In[-4])+INT(In[4]))
				              +   1591*(INT(In[-3])+INT(In[3]))
				              +   1662*(INT(In[-2])+INT(In[2]))
				              +   1705*(INT(In[-1])+INT(In[1]))
				              +   1720*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num74(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-30),-30,30,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num75 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num75(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   1*(INT(In[-30])+INT(In[30]))
				              +   1*(INT(In[-29])+INT(In[29]))
				              +   2*(INT(In[-28])+INT(In[28]))
				              +   4*(INT(In[-27])+INT(In[27]))
				              +   6*(INT(In[-26])+INT(In[26]))
				              +   9*(INT(In[-25])+INT(In[25]))
				              +   13*(INT(In[-24])+INT(In[24]))
				              +   20*(INT(In[-23])+INT(In[23]))
				              +   29*(INT(In[-22])+INT(In[22]))
				              +   41*(INT(In[-21])+INT(In[21]))
				              +   58*(INT(In[-20])+INT(In[20]))
				              +   81*(INT(In[-19])+INT(In[19]))
				              +   110*(INT(In[-18])+INT(In[18]))
				              +   148*(INT(In[-17])+INT(In[17]))
				              +   196*(INT(In[-16])+INT(In[16]))
				              +   255*(INT(In[-15])+INT(In[15]))
				              +   325*(INT(In[-14])+INT(In[14]))
				              +   408*(INT(In[-13])+INT(In[13]))
				              +   504*(INT(In[-12])+INT(In[12]))
				              +   612*(INT(In[-11])+INT(In[11]))
				              +   731*(INT(In[-10])+INT(In[10]))
				              +   857*(INT(In[-9])+INT(In[9]))
				              +   990*(INT(In[-8])+INT(In[8]))
				              +   1123*(INT(In[-7])+INT(In[7]))
				              +   1253*(INT(In[-6])+INT(In[6]))
				              +   1375*(INT(In[-5])+INT(In[5]))
				              +   1483*(INT(In[-4])+INT(In[4]))
				              +   1574*(INT(In[-3])+INT(In[3]))
				              +   1641*(INT(In[-2])+INT(In[2]))
				              +   1684*(INT(In[-1])+INT(In[1]))
				              +   1698*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num75(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-31),-31,31,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num76 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num76(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   1*(INT(In[-30])+INT(In[30]))
				              +   2*(INT(In[-29])+INT(In[29]))
				              +   3*(INT(In[-28])+INT(In[28]))
				              +   4*(INT(In[-27])+INT(In[27]))
				              +   6*(INT(In[-26])+INT(In[26]))
				              +   10*(INT(In[-25])+INT(In[25]))
				              +   15*(INT(In[-24])+INT(In[24]))
				              +   22*(INT(In[-23])+INT(In[23]))
				              +   31*(INT(In[-22])+INT(In[22]))
				              +   45*(INT(In[-21])+INT(In[21]))
				              +   63*(INT(In[-20])+INT(In[20]))
				              +   86*(INT(In[-19])+INT(In[19]))
				              +   117*(INT(In[-18])+INT(In[18]))
				              +   156*(INT(In[-17])+INT(In[17]))
				              +   204*(INT(In[-16])+INT(In[16]))
				              +   264*(INT(In[-15])+INT(In[15]))
				              +   335*(INT(In[-14])+INT(In[14]))
				              +   418*(INT(In[-13])+INT(In[13]))
				              +   513*(INT(In[-12])+INT(In[12]))
				              +   620*(INT(In[-11])+INT(In[11]))
				              +   737*(INT(In[-10])+INT(In[10]))
				              +   861*(INT(In[-9])+INT(In[9]))
				              +   990*(INT(In[-8])+INT(In[8]))
				              +   1120*(INT(In[-7])+INT(In[7]))
				              +   1247*(INT(In[-6])+INT(In[6]))
				              +   1365*(INT(In[-5])+INT(In[5]))
				              +   1469*(INT(In[-4])+INT(In[4]))
				              +   1557*(INT(In[-3])+INT(In[3]))
				              +   1622*(INT(In[-2])+INT(In[2]))
				              +   1662*(INT(In[-1])+INT(In[1]))
				              +   1676*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num76(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-31),-31,31,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num77 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num77(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   1*(INT(In[-30])+INT(In[30]))
				              +   2*(INT(In[-29])+INT(In[29]))
				              +   3*(INT(In[-28])+INT(In[28]))
				              +   5*(INT(In[-27])+INT(In[27]))
				              +   7*(INT(In[-26])+INT(In[26]))
				              +   11*(INT(In[-25])+INT(In[25]))
				              +   16*(INT(In[-24])+INT(In[24]))
				              +   24*(INT(In[-23])+INT(In[23]))
				              +   34*(INT(In[-22])+INT(In[22]))
				              +   48*(INT(In[-21])+INT(In[21]))
				              +   67*(INT(In[-20])+INT(In[20]))
				              +   92*(INT(In[-19])+INT(In[19]))
				              +   123*(INT(In[-18])+INT(In[18]))
				              +   163*(INT(In[-17])+INT(In[17]))
				              +   213*(INT(In[-16])+INT(In[16]))
				              +   273*(INT(In[-15])+INT(In[15]))
				              +   344*(INT(In[-14])+INT(In[14]))
				              +   427*(INT(In[-13])+INT(In[13]))
				              +   522*(INT(In[-12])+INT(In[12]))
				              +   628*(INT(In[-11])+INT(In[11]))
				              +   743*(INT(In[-10])+INT(In[10]))
				              +   865*(INT(In[-9])+INT(In[9]))
				              +   991*(INT(In[-8])+INT(In[8]))
				              +   1118*(INT(In[-7])+INT(In[7]))
				              +   1240*(INT(In[-6])+INT(In[6]))
				              +   1354*(INT(In[-5])+INT(In[5]))
				              +   1456*(INT(In[-4])+INT(In[4]))
				              +   1540*(INT(In[-3])+INT(In[3]))
				              +   1603*(INT(In[-2])+INT(In[2]))
				              +   1642*(INT(In[-1])+INT(In[1]))
				              +   1656*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num77(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-31),-31,31,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num78 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num78(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   2*(INT(In[-30])+INT(In[30]))
				              +   2*(INT(In[-29])+INT(In[29]))
				              +   4*(INT(In[-28])+INT(In[28]))
				              +   6*(INT(In[-27])+INT(In[27]))
				              +   8*(INT(In[-26])+INT(In[26]))
				              +   12*(INT(In[-25])+INT(In[25]))
				              +   18*(INT(In[-24])+INT(In[24]))
				              +   26*(INT(In[-23])+INT(In[23]))
				              +   37*(INT(In[-22])+INT(In[22]))
				              +   52*(INT(In[-21])+INT(In[21]))
				              +   72*(INT(In[-20])+INT(In[20]))
				              +   97*(INT(In[-19])+INT(In[19]))
				              +   130*(INT(In[-18])+INT(In[18]))
				              +   171*(INT(In[-17])+INT(In[17]))
				              +   221*(INT(In[-16])+INT(In[16]))
				              +   282*(INT(In[-15])+INT(In[15]))
				              +   353*(INT(In[-14])+INT(In[14]))
				              +   436*(INT(In[-13])+INT(In[13]))
				              +   531*(INT(In[-12])+INT(In[12]))
				              +   635*(INT(In[-11])+INT(In[11]))
				              +   748*(INT(In[-10])+INT(In[10]))
				              +   868*(INT(In[-9])+INT(In[9]))
				              +   991*(INT(In[-8])+INT(In[8]))
				              +   1114*(INT(In[-7])+INT(In[7]))
				              +   1234*(INT(In[-6])+INT(In[6]))
				              +   1344*(INT(In[-5])+INT(In[5]))
				              +   1442*(INT(In[-4])+INT(In[4]))
				              +   1523*(INT(In[-3])+INT(In[3]))
				              +   1584*(INT(In[-2])+INT(In[2]))
				              +   1622*(INT(In[-1])+INT(In[1]))
				              +   1634*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num78(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-32),-32,32,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num79 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num79(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   2*(INT(In[-30])+INT(In[30]))
				              +   3*(INT(In[-29])+INT(In[29]))
				              +   4*(INT(In[-28])+INT(In[28]))
				              +   6*(INT(In[-27])+INT(In[27]))
				              +   9*(INT(In[-26])+INT(In[26]))
				              +   14*(INT(In[-25])+INT(In[25]))
				              +   20*(INT(In[-24])+INT(In[24]))
				              +   29*(INT(In[-23])+INT(In[23]))
				              +   40*(INT(In[-22])+INT(In[22]))
				              +   56*(INT(In[-21])+INT(In[21]))
				              +   77*(INT(In[-20])+INT(In[20]))
				              +   103*(INT(In[-19])+INT(In[19]))
				              +   137*(INT(In[-18])+INT(In[18]))
				              +   178*(INT(In[-17])+INT(In[17]))
				              +   229*(INT(In[-16])+INT(In[16]))
				              +   290*(INT(In[-15])+INT(In[15]))
				              +   362*(INT(In[-14])+INT(In[14]))
				              +   445*(INT(In[-13])+INT(In[13]))
				              +   539*(INT(In[-12])+INT(In[12]))
				              +   642*(INT(In[-11])+INT(In[11]))
				              +   753*(INT(In[-10])+INT(In[10]))
				              +   871*(INT(In[-9])+INT(In[9]))
				              +   991*(INT(In[-8])+INT(In[8]))
				              +   1111*(INT(In[-7])+INT(In[7]))
				              +   1227*(INT(In[-6])+INT(In[6]))
				              +   1334*(INT(In[-5])+INT(In[5]))
				              +   1429*(INT(In[-4])+INT(In[4]))
				              +   1507*(INT(In[-3])+INT(In[3]))
				              +   1565*(INT(In[-2])+INT(In[2]))
				              +   1602*(INT(In[-1])+INT(In[1]))
				              +   1614*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num79(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-32),-32,32,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num80 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num80(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   2*(INT(In[-30])+INT(In[30]))
				              +   3*(INT(In[-29])+INT(In[29]))
				              +   5*(INT(In[-28])+INT(In[28]))
				              +   7*(INT(In[-27])+INT(In[27]))
				              +   10*(INT(In[-26])+INT(In[26]))
				              +   15*(INT(In[-25])+INT(In[25]))
				              +   22*(INT(In[-24])+INT(In[24]))
				              +   31*(INT(In[-23])+INT(In[23]))
				              +   44*(INT(In[-22])+INT(In[22]))
				              +   60*(INT(In[-21])+INT(In[21]))
				              +   81*(INT(In[-20])+INT(In[20]))
				              +   109*(INT(In[-19])+INT(In[19]))
				              +   143*(INT(In[-18])+INT(In[18]))
				              +   186*(INT(In[-17])+INT(In[17]))
				              +   238*(INT(In[-16])+INT(In[16]))
				              +   299*(INT(In[-15])+INT(In[15]))
				              +   371*(INT(In[-14])+INT(In[14]))
				              +   454*(INT(In[-13])+INT(In[13]))
				              +   546*(INT(In[-12])+INT(In[12]))
				              +   648*(INT(In[-11])+INT(In[11]))
				              +   758*(INT(In[-10])+INT(In[10]))
				              +   873*(INT(In[-9])+INT(In[9]))
				              +   991*(INT(In[-8])+INT(In[8]))
				              +   1107*(INT(In[-7])+INT(In[7]))
				              +   1220*(INT(In[-6])+INT(In[6]))
				              +   1324*(INT(In[-5])+INT(In[5]))
				              +   1415*(INT(In[-4])+INT(In[4]))
				              +   1491*(INT(In[-3])+INT(In[3]))
				              +   1548*(INT(In[-2])+INT(In[2]))
				              +   1583*(INT(In[-1])+INT(In[1]))
				              +   1594*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num80(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-33),-33,33,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num81 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num81(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   1*(INT(In[-31])+INT(In[31]))
				              +   2*(INT(In[-30])+INT(In[30]))
				              +   4*(INT(In[-29])+INT(In[29]))
				              +   5*(INT(In[-28])+INT(In[28]))
				              +   8*(INT(In[-27])+INT(In[27]))
				              +   12*(INT(In[-26])+INT(In[26]))
				              +   17*(INT(In[-25])+INT(In[25]))
				              +   24*(INT(In[-24])+INT(In[24]))
				              +   34*(INT(In[-23])+INT(In[23]))
				              +   47*(INT(In[-22])+INT(In[22]))
				              +   64*(INT(In[-21])+INT(In[21]))
				              +   86*(INT(In[-20])+INT(In[20]))
				              +   115*(INT(In[-19])+INT(In[19]))
				              +   150*(INT(In[-18])+INT(In[18]))
				              +   193*(INT(In[-17])+INT(In[17]))
				              +   246*(INT(In[-16])+INT(In[16]))
				              +   308*(INT(In[-15])+INT(In[15]))
				              +   380*(INT(In[-14])+INT(In[14]))
				              +   462*(INT(In[-13])+INT(In[13]))
				              +   554*(INT(In[-12])+INT(In[12]))
				              +   654*(INT(In[-11])+INT(In[11]))
				              +   762*(INT(In[-10])+INT(In[10]))
				              +   875*(INT(In[-9])+INT(In[9]))
				              +   990*(INT(In[-8])+INT(In[8]))
				              +   1104*(INT(In[-7])+INT(In[7]))
				              +   1213*(INT(In[-6])+INT(In[6]))
				              +   1314*(INT(In[-5])+INT(In[5]))
				              +   1402*(INT(In[-4])+INT(In[4]))
				              +   1475*(INT(In[-3])+INT(In[3]))
				              +   1530*(INT(In[-2])+INT(In[2]))
				              +   1564*(INT(In[-1])+INT(In[1]))
				              +   1574*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num81(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-33),-33,33,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num82 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num82(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   2*(INT(In[-31])+INT(In[31]))
				              +   3*(INT(In[-30])+INT(In[30]))
				              +   4*(INT(In[-29])+INT(In[29]))
				              +   6*(INT(In[-28])+INT(In[28]))
				              +   9*(INT(In[-27])+INT(In[27]))
				              +   13*(INT(In[-26])+INT(In[26]))
				              +   19*(INT(In[-25])+INT(In[25]))
				              +   26*(INT(In[-24])+INT(In[24]))
				              +   37*(INT(In[-23])+INT(In[23]))
				              +   50*(INT(In[-22])+INT(In[22]))
				              +   68*(INT(In[-21])+INT(In[21]))
				              +   91*(INT(In[-20])+INT(In[20]))
				              +   120*(INT(In[-19])+INT(In[19]))
				              +   157*(INT(In[-18])+INT(In[18]))
				              +   201*(INT(In[-17])+INT(In[17]))
				              +   254*(INT(In[-16])+INT(In[16]))
				              +   316*(INT(In[-15])+INT(In[15]))
				              +   388*(INT(In[-14])+INT(In[14]))
				              +   470*(INT(In[-13])+INT(In[13]))
				              +   561*(INT(In[-12])+INT(In[12]))
				              +   660*(INT(In[-11])+INT(In[11]))
				              +   766*(INT(In[-10])+INT(In[10]))
				              +   877*(INT(In[-9])+INT(In[9]))
				              +   989*(INT(In[-8])+INT(In[8]))
				              +   1100*(INT(In[-7])+INT(In[7]))
				              +   1206*(INT(In[-6])+INT(In[6]))
				              +   1304*(INT(In[-5])+INT(In[5]))
				              +   1389*(INT(In[-4])+INT(In[4]))
				              +   1460*(INT(In[-3])+INT(In[3]))
				              +   1513*(INT(In[-2])+INT(In[2]))
				              +   1545*(INT(In[-1])+INT(In[1]))
				              +   1556*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num82(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-33),-33,33,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num83 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num83(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   1*(INT(In[-32])+INT(In[32]))
				              +   2*(INT(In[-31])+INT(In[31]))
				              +   3*(INT(In[-30])+INT(In[30]))
				              +   5*(INT(In[-29])+INT(In[29]))
				              +   7*(INT(In[-28])+INT(In[28]))
				              +   10*(INT(In[-27])+INT(In[27]))
				              +   14*(INT(In[-26])+INT(In[26]))
				              +   20*(INT(In[-25])+INT(In[25]))
				              +   29*(INT(In[-24])+INT(In[24]))
				              +   39*(INT(In[-23])+INT(In[23]))
				              +   54*(INT(In[-22])+INT(In[22]))
				              +   73*(INT(In[-21])+INT(In[21]))
				              +   97*(INT(In[-20])+INT(In[20]))
				              +   126*(INT(In[-19])+INT(In[19]))
				              +   163*(INT(In[-18])+INT(In[18]))
				              +   208*(INT(In[-17])+INT(In[17]))
				              +   261*(INT(In[-16])+INT(In[16]))
				              +   324*(INT(In[-15])+INT(In[15]))
				              +   396*(INT(In[-14])+INT(In[14]))
				              +   478*(INT(In[-13])+INT(In[13]))
				              +   568*(INT(In[-12])+INT(In[12]))
				              +   666*(INT(In[-11])+INT(In[11]))
				              +   770*(INT(In[-10])+INT(In[10]))
				              +   878*(INT(In[-9])+INT(In[9]))
				              +   988*(INT(In[-8])+INT(In[8]))
				              +   1096*(INT(In[-7])+INT(In[7]))
				              +   1199*(INT(In[-6])+INT(In[6]))
				              +   1294*(INT(In[-5])+INT(In[5]))
				              +   1377*(INT(In[-4])+INT(In[4]))
				              +   1445*(INT(In[-3])+INT(In[3]))
				              +   1496*(INT(In[-2])+INT(In[2]))
				              +   1527*(INT(In[-1])+INT(In[1]))
				              +   1538*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num83(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-33),-33,33,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num84 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num84(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   2*(INT(In[-32])+INT(In[32]))
				              +   2*(INT(In[-31])+INT(In[31]))
				              +   3*(INT(In[-30])+INT(In[30]))
				              +   5*(INT(In[-29])+INT(In[29]))
				              +   8*(INT(In[-28])+INT(In[28]))
				              +   11*(INT(In[-27])+INT(In[27]))
				              +   16*(INT(In[-26])+INT(In[26]))
				              +   22*(INT(In[-25])+INT(In[25]))
				              +   31*(INT(In[-24])+INT(In[24]))
				              +   43*(INT(In[-23])+INT(In[23]))
				              +   58*(INT(In[-22])+INT(In[22]))
				              +   77*(INT(In[-21])+INT(In[21]))
				              +   102*(INT(In[-20])+INT(In[20]))
				              +   132*(INT(In[-19])+INT(In[19]))
				              +   170*(INT(In[-18])+INT(In[18]))
				              +   215*(INT(In[-17])+INT(In[17]))
				              +   269*(INT(In[-16])+INT(In[16]))
				              +   332*(INT(In[-15])+INT(In[15]))
				              +   404*(INT(In[-14])+INT(In[14]))
				              +   485*(INT(In[-13])+INT(In[13]))
				              +   574*(INT(In[-12])+INT(In[12]))
				              +   671*(INT(In[-11])+INT(In[11]))
				              +   773*(INT(In[-10])+INT(In[10]))
				              +   879*(INT(In[-9])+INT(In[9]))
				              +   986*(INT(In[-8])+INT(In[8]))
				              +   1092*(INT(In[-7])+INT(In[7]))
				              +   1192*(INT(In[-6])+INT(In[6]))
				              +   1284*(INT(In[-5])+INT(In[5]))
				              +   1364*(INT(In[-4])+INT(In[4]))
				              +   1430*(INT(In[-3])+INT(In[3]))
				              +   1480*(INT(In[-2])+INT(In[2]))
				              +   1510*(INT(In[-1])+INT(In[1]))
				              +   1520*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num84(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-34),-34,34,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num85 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num85(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   2*(INT(In[-32])+INT(In[32]))
				              +   3*(INT(In[-31])+INT(In[31]))
				              +   4*(INT(In[-30])+INT(In[30]))
				              +   6*(INT(In[-29])+INT(In[29]))
				              +   8*(INT(In[-28])+INT(In[28]))
				              +   12*(INT(In[-27])+INT(In[27]))
				              +   17*(INT(In[-26])+INT(In[26]))
				              +   24*(INT(In[-25])+INT(In[25]))
				              +   33*(INT(In[-24])+INT(In[24]))
				              +   46*(INT(In[-23])+INT(In[23]))
				              +   61*(INT(In[-22])+INT(In[22]))
				              +   82*(INT(In[-21])+INT(In[21]))
				              +   107*(INT(In[-20])+INT(In[20]))
				              +   138*(INT(In[-19])+INT(In[19]))
				              +   177*(INT(In[-18])+INT(In[18]))
				              +   223*(INT(In[-17])+INT(In[17]))
				              +   277*(INT(In[-16])+INT(In[16]))
				              +   340*(INT(In[-15])+INT(In[15]))
				              +   412*(INT(In[-14])+INT(In[14]))
				              +   492*(INT(In[-13])+INT(In[13]))
				              +   580*(INT(In[-12])+INT(In[12]))
				              +   676*(INT(In[-11])+INT(In[11]))
				              +   776*(INT(In[-10])+INT(In[10]))
				              +   880*(INT(In[-9])+INT(In[9]))
				              +   985*(INT(In[-8])+INT(In[8]))
				              +   1087*(INT(In[-7])+INT(In[7]))
				              +   1185*(INT(In[-6])+INT(In[6]))
				              +   1274*(INT(In[-5])+INT(In[5]))
				              +   1352*(INT(In[-4])+INT(In[4]))
				              +   1416*(INT(In[-3])+INT(In[3]))
				              +   1463*(INT(In[-2])+INT(In[2]))
				              +   1493*(INT(In[-1])+INT(In[1]))
				              +   1502*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num85(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-34),-34,34,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num86 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num86(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   1*(INT(In[-33])+INT(In[33]))
				              +   2*(INT(In[-32])+INT(In[32]))
				              +   3*(INT(In[-31])+INT(In[31]))
				              +   4*(INT(In[-30])+INT(In[30]))
				              +   7*(INT(In[-29])+INT(In[29]))
				              +   9*(INT(In[-28])+INT(In[28]))
				              +   13*(INT(In[-27])+INT(In[27]))
				              +   19*(INT(In[-26])+INT(In[26]))
				              +   26*(INT(In[-25])+INT(In[25]))
				              +   36*(INT(In[-24])+INT(In[24]))
				              +   49*(INT(In[-23])+INT(In[23]))
				              +   65*(INT(In[-22])+INT(In[22]))
				              +   86*(INT(In[-21])+INT(In[21]))
				              +   112*(INT(In[-20])+INT(In[20]))
				              +   144*(INT(In[-19])+INT(In[19]))
				              +   183*(INT(In[-18])+INT(In[18]))
				              +   230*(INT(In[-17])+INT(In[17]))
				              +   285*(INT(In[-16])+INT(In[16]))
				              +   348*(INT(In[-15])+INT(In[15]))
				              +   419*(INT(In[-14])+INT(In[14]))
				              +   499*(INT(In[-13])+INT(In[13]))
				              +   586*(INT(In[-12])+INT(In[12]))
				              +   680*(INT(In[-11])+INT(In[11]))
				              +   779*(INT(In[-10])+INT(In[10]))
				              +   881*(INT(In[-9])+INT(In[9]))
				              +   983*(INT(In[-8])+INT(In[8]))
				              +   1083*(INT(In[-7])+INT(In[7]))
				              +   1177*(INT(In[-6])+INT(In[6]))
				              +   1264*(INT(In[-5])+INT(In[5]))
				              +   1340*(INT(In[-4])+INT(In[4]))
				              +   1402*(INT(In[-3])+INT(In[3]))
				              +   1448*(INT(In[-2])+INT(In[2]))
				              +   1476*(INT(In[-1])+INT(In[1]))
				              +   1486*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num86(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-35),-35,35,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num87 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num87(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   2*(INT(In[-33])+INT(In[33]))
				              +   2*(INT(In[-32])+INT(In[32]))
				              +   3*(INT(In[-31])+INT(In[31]))
				              +   5*(INT(In[-30])+INT(In[30]))
				              +   7*(INT(In[-29])+INT(In[29]))
				              +   10*(INT(In[-28])+INT(In[28]))
				              +   15*(INT(In[-27])+INT(In[27]))
				              +   21*(INT(In[-26])+INT(In[26]))
				              +   28*(INT(In[-25])+INT(In[25]))
				              +   39*(INT(In[-24])+INT(In[24]))
				              +   52*(INT(In[-23])+INT(In[23]))
				              +   69*(INT(In[-22])+INT(In[22]))
				              +   91*(INT(In[-21])+INT(In[21]))
				              +   118*(INT(In[-20])+INT(In[20]))
				              +   150*(INT(In[-19])+INT(In[19]))
				              +   190*(INT(In[-18])+INT(In[18]))
				              +   237*(INT(In[-17])+INT(In[17]))
				              +   292*(INT(In[-16])+INT(In[16]))
				              +   355*(INT(In[-15])+INT(In[15]))
				              +   426*(INT(In[-14])+INT(In[14]))
				              +   506*(INT(In[-13])+INT(In[13]))
				              +   592*(INT(In[-12])+INT(In[12]))
				              +   684*(INT(In[-11])+INT(In[11]))
				              +   781*(INT(In[-10])+INT(In[10]))
				              +   881*(INT(In[-9])+INT(In[9]))
				              +   981*(INT(In[-8])+INT(In[8]))
				              +   1078*(INT(In[-7])+INT(In[7]))
				              +   1170*(INT(In[-6])+INT(In[6]))
				              +   1254*(INT(In[-5])+INT(In[5]))
				              +   1328*(INT(In[-4])+INT(In[4]))
				              +   1388*(INT(In[-3])+INT(In[3]))
				              +   1432*(INT(In[-2])+INT(In[2]))
				              +   1460*(INT(In[-1])+INT(In[1]))
				              +   1470*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num87(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-35),-35,35,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num88 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num88(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   2*(INT(In[-33])+INT(In[33]))
				              +   3*(INT(In[-32])+INT(In[32]))
				              +   4*(INT(In[-31])+INT(In[31]))
				              +   6*(INT(In[-30])+INT(In[30]))
				              +   8*(INT(In[-29])+INT(In[29]))
				              +   11*(INT(In[-28])+INT(In[28]))
				              +   16*(INT(In[-27])+INT(In[27]))
				              +   22*(INT(In[-26])+INT(In[26]))
				              +   31*(INT(In[-25])+INT(In[25]))
				              +   42*(INT(In[-24])+INT(In[24]))
				              +   55*(INT(In[-23])+INT(In[23]))
				              +   73*(INT(In[-22])+INT(In[22]))
				              +   95*(INT(In[-21])+INT(In[21]))
				              +   123*(INT(In[-20])+INT(In[20]))
				              +   156*(INT(In[-19])+INT(In[19]))
				              +   197*(INT(In[-18])+INT(In[18]))
				              +   244*(INT(In[-17])+INT(In[17]))
				              +   299*(INT(In[-16])+INT(In[16]))
				              +   362*(INT(In[-15])+INT(In[15]))
				              +   433*(INT(In[-14])+INT(In[14]))
				              +   512*(INT(In[-13])+INT(In[13]))
				              +   597*(INT(In[-12])+INT(In[12]))
				              +   688*(INT(In[-11])+INT(In[11]))
				              +   784*(INT(In[-10])+INT(In[10]))
				              +   881*(INT(In[-9])+INT(In[9]))
				              +   979*(INT(In[-8])+INT(In[8]))
				              +   1073*(INT(In[-7])+INT(In[7]))
				              +   1163*(INT(In[-6])+INT(In[6]))
				              +   1245*(INT(In[-5])+INT(In[5]))
				              +   1316*(INT(In[-4])+INT(In[4]))
				              +   1374*(INT(In[-3])+INT(In[3]))
				              +   1417*(INT(In[-2])+INT(In[2]))
				              +   1444*(INT(In[-1])+INT(In[1]))
				              +   1454*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num88(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-35),-35,35,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num89 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num89(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   1*(INT(In[-34])+INT(In[34]))
				              +   2*(INT(In[-33])+INT(In[33]))
				              +   3*(INT(In[-32])+INT(In[32]))
				              +   4*(INT(In[-31])+INT(In[31]))
				              +   6*(INT(In[-30])+INT(In[30]))
				              +   9*(INT(In[-29])+INT(In[29]))
				              +   13*(INT(In[-28])+INT(In[28]))
				              +   18*(INT(In[-27])+INT(In[27]))
				              +   24*(INT(In[-26])+INT(In[26]))
				              +   33*(INT(In[-25])+INT(In[25]))
				              +   44*(INT(In[-24])+INT(In[24]))
				              +   59*(INT(In[-23])+INT(In[23]))
				              +   77*(INT(In[-22])+INT(In[22]))
				              +   100*(INT(In[-21])+INT(In[21]))
				              +   128*(INT(In[-20])+INT(In[20]))
				              +   163*(INT(In[-19])+INT(In[19]))
				              +   203*(INT(In[-18])+INT(In[18]))
				              +   251*(INT(In[-17])+INT(In[17]))
				              +   306*(INT(In[-16])+INT(In[16]))
				              +   369*(INT(In[-15])+INT(In[15]))
				              +   440*(INT(In[-14])+INT(In[14]))
				              +   518*(INT(In[-13])+INT(In[13]))
				              +   602*(INT(In[-12])+INT(In[12]))
				              +   692*(INT(In[-11])+INT(In[11]))
				              +   786*(INT(In[-10])+INT(In[10]))
				              +   881*(INT(In[-9])+INT(In[9]))
				              +   976*(INT(In[-8])+INT(In[8]))
				              +   1069*(INT(In[-7])+INT(In[7]))
				              +   1156*(INT(In[-6])+INT(In[6]))
				              +   1235*(INT(In[-5])+INT(In[5]))
				              +   1304*(INT(In[-4])+INT(In[4]))
				              +   1361*(INT(In[-3])+INT(In[3]))
				              +   1402*(INT(In[-2])+INT(In[2]))
				              +   1428*(INT(In[-1])+INT(In[1]))
				              +   1438*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num89(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-36),-36,36,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num90 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num90(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   2*(INT(In[-34])+INT(In[34]))
				              +   2*(INT(In[-33])+INT(In[33]))
				              +   3*(INT(In[-32])+INT(In[32]))
				              +   5*(INT(In[-31])+INT(In[31]))
				              +   7*(INT(In[-30])+INT(In[30]))
				              +   10*(INT(In[-29])+INT(In[29]))
				              +   14*(INT(In[-28])+INT(In[28]))
				              +   19*(INT(In[-27])+INT(In[27]))
				              +   26*(INT(In[-26])+INT(In[26]))
				              +   35*(INT(In[-25])+INT(In[25]))
				              +   47*(INT(In[-24])+INT(In[24]))
				              +   62*(INT(In[-23])+INT(In[23]))
				              +   81*(INT(In[-22])+INT(In[22]))
				              +   105*(INT(In[-21])+INT(In[21]))
				              +   134*(INT(In[-20])+INT(In[20]))
				              +   168*(INT(In[-19])+INT(In[19]))
				              +   210*(INT(In[-18])+INT(In[18]))
				              +   258*(INT(In[-17])+INT(In[17]))
				              +   313*(INT(In[-16])+INT(In[16]))
				              +   376*(INT(In[-15])+INT(In[15]))
				              +   446*(INT(In[-14])+INT(In[14]))
				              +   524*(INT(In[-13])+INT(In[13]))
				              +   607*(INT(In[-12])+INT(In[12]))
				              +   695*(INT(In[-11])+INT(In[11]))
				              +   787*(INT(In[-10])+INT(In[10]))
				              +   881*(INT(In[-9])+INT(In[9]))
				              +   974*(INT(In[-8])+INT(In[8]))
				              +   1064*(INT(In[-7])+INT(In[7]))
				              +   1149*(INT(In[-6])+INT(In[6]))
				              +   1226*(INT(In[-5])+INT(In[5]))
				              +   1293*(INT(In[-4])+INT(In[4]))
				              +   1347*(INT(In[-3])+INT(In[3]))
				              +   1388*(INT(In[-2])+INT(In[2]))
				              +   1413*(INT(In[-1])+INT(In[1]))
				              +   1422*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num90(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-36),-36,36,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num91 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num91(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   2*(INT(In[-34])+INT(In[34]))
				              +   3*(INT(In[-33])+INT(In[33]))
				              +   4*(INT(In[-32])+INT(In[32]))
				              +   5*(INT(In[-31])+INT(In[31]))
				              +   8*(INT(In[-30])+INT(In[30]))
				              +   11*(INT(In[-29])+INT(In[29]))
				              +   15*(INT(In[-28])+INT(In[28]))
				              +   21*(INT(In[-27])+INT(In[27]))
				              +   28*(INT(In[-26])+INT(In[26]))
				              +   38*(INT(In[-25])+INT(In[25]))
				              +   50*(INT(In[-24])+INT(In[24]))
				              +   66*(INT(In[-23])+INT(In[23]))
				              +   86*(INT(In[-22])+INT(In[22]))
				              +   110*(INT(In[-21])+INT(In[21]))
				              +   139*(INT(In[-20])+INT(In[20]))
				              +   174*(INT(In[-19])+INT(In[19]))
				              +   216*(INT(In[-18])+INT(In[18]))
				              +   264*(INT(In[-17])+INT(In[17]))
				              +   320*(INT(In[-16])+INT(In[16]))
				              +   383*(INT(In[-15])+INT(In[15]))
				              +   453*(INT(In[-14])+INT(In[14]))
				              +   529*(INT(In[-13])+INT(In[13]))
				              +   611*(INT(In[-12])+INT(In[12]))
				              +   698*(INT(In[-11])+INT(In[11]))
				              +   789*(INT(In[-10])+INT(In[10]))
				              +   880*(INT(In[-9])+INT(In[9]))
				              +   971*(INT(In[-8])+INT(In[8]))
				              +   1059*(INT(In[-7])+INT(In[7]))
				              +   1142*(INT(In[-6])+INT(In[6]))
				              +   1217*(INT(In[-5])+INT(In[5]))
				              +   1281*(INT(In[-4])+INT(In[4]))
				              +   1334*(INT(In[-3])+INT(In[3]))
				              +   1374*(INT(In[-2])+INT(In[2]))
				              +   1398*(INT(In[-1])+INT(In[1]))
				              +   1406*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num91(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-36),-36,36,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num92 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num92(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   1*(INT(In[-35])+INT(In[35]))
				              +   2*(INT(In[-34])+INT(In[34]))
				              +   3*(INT(In[-33])+INT(In[33]))
				              +   4*(INT(In[-32])+INT(In[32]))
				              +   6*(INT(In[-31])+INT(In[31]))
				              +   9*(INT(In[-30])+INT(In[30]))
				              +   12*(INT(In[-29])+INT(In[29]))
				              +   16*(INT(In[-28])+INT(In[28]))
				              +   22*(INT(In[-27])+INT(In[27]))
				              +   30*(INT(In[-26])+INT(In[26]))
				              +   41*(INT(In[-25])+INT(In[25]))
				              +   53*(INT(In[-24])+INT(In[24]))
				              +   70*(INT(In[-23])+INT(In[23]))
				              +   90*(INT(In[-22])+INT(In[22]))
				              +   115*(INT(In[-21])+INT(In[21]))
				              +   145*(INT(In[-20])+INT(In[20]))
				              +   180*(INT(In[-19])+INT(In[19]))
				              +   222*(INT(In[-18])+INT(In[18]))
				              +   271*(INT(In[-17])+INT(In[17]))
				              +   327*(INT(In[-16])+INT(In[16]))
				              +   389*(INT(In[-15])+INT(In[15]))
				              +   459*(INT(In[-14])+INT(In[14]))
				              +   535*(INT(In[-13])+INT(In[13]))
				              +   616*(INT(In[-12])+INT(In[12]))
				              +   701*(INT(In[-11])+INT(In[11]))
				              +   790*(INT(In[-10])+INT(In[10]))
				              +   879*(INT(In[-9])+INT(In[9]))
				              +   968*(INT(In[-8])+INT(In[8]))
				              +   1054*(INT(In[-7])+INT(In[7]))
				              +   1134*(INT(In[-6])+INT(In[6]))
				              +   1207*(INT(In[-5])+INT(In[5]))
				              +   1270*(INT(In[-4])+INT(In[4]))
				              +   1322*(INT(In[-3])+INT(In[3]))
				              +   1360*(INT(In[-2])+INT(In[2]))
				              +   1383*(INT(In[-1])+INT(In[1]))
				              +   1392*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num92(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-37),-37,37,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num93 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num93(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   2*(INT(In[-35])+INT(In[35]))
				              +   2*(INT(In[-34])+INT(In[34]))
				              +   3*(INT(In[-33])+INT(In[33]))
				              +   5*(INT(In[-32])+INT(In[32]))
				              +   7*(INT(In[-31])+INT(In[31]))
				              +   9*(INT(In[-30])+INT(In[30]))
				              +   13*(INT(In[-29])+INT(In[29]))
				              +   18*(INT(In[-28])+INT(In[28]))
				              +   24*(INT(In[-27])+INT(In[27]))
				              +   32*(INT(In[-26])+INT(In[26]))
				              +   43*(INT(In[-25])+INT(In[25]))
				              +   57*(INT(In[-24])+INT(In[24]))
				              +   73*(INT(In[-23])+INT(In[23]))
				              +   94*(INT(In[-22])+INT(In[22]))
				              +   120*(INT(In[-21])+INT(In[21]))
				              +   150*(INT(In[-20])+INT(In[20]))
				              +   186*(INT(In[-19])+INT(In[19]))
				              +   229*(INT(In[-18])+INT(In[18]))
				              +   278*(INT(In[-17])+INT(In[17]))
				              +   333*(INT(In[-16])+INT(In[16]))
				              +   396*(INT(In[-15])+INT(In[15]))
				              +   465*(INT(In[-14])+INT(In[14]))
				              +   540*(INT(In[-13])+INT(In[13]))
				              +   620*(INT(In[-12])+INT(In[12]))
				              +   704*(INT(In[-11])+INT(In[11]))
				              +   791*(INT(In[-10])+INT(In[10]))
				              +   879*(INT(In[-9])+INT(In[9]))
				              +   965*(INT(In[-8])+INT(In[8]))
				              +   1049*(INT(In[-7])+INT(In[7]))
				              +   1127*(INT(In[-6])+INT(In[6]))
				              +   1198*(INT(In[-5])+INT(In[5]))
				              +   1259*(INT(In[-4])+INT(In[4]))
				              +   1309*(INT(In[-3])+INT(In[3]))
				              +   1346*(INT(In[-2])+INT(In[2]))
				              +   1368*(INT(In[-1])+INT(In[1]))
				              +   1376*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num93(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-37),-37,37,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num94 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num94(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-38])+INT(In[38]))
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   2*(INT(In[-35])+INT(In[35]))
				              +   3*(INT(In[-34])+INT(In[34]))
				              +   4*(INT(In[-33])+INT(In[33]))
				              +   5*(INT(In[-32])+INT(In[32]))
				              +   7*(INT(In[-31])+INT(In[31]))
				              +   10*(INT(In[-30])+INT(In[30]))
				              +   14*(INT(In[-29])+INT(In[29]))
				              +   19*(INT(In[-28])+INT(In[28]))
				              +   26*(INT(In[-27])+INT(In[27]))
				              +   35*(INT(In[-26])+INT(In[26]))
				              +   46*(INT(In[-25])+INT(In[25]))
				              +   60*(INT(In[-24])+INT(In[24]))
				              +   77*(INT(In[-23])+INT(In[23]))
				              +   99*(INT(In[-22])+INT(In[22]))
				              +   124*(INT(In[-21])+INT(In[21]))
				              +   155*(INT(In[-20])+INT(In[20]))
				              +   192*(INT(In[-19])+INT(In[19]))
				              +   235*(INT(In[-18])+INT(In[18]))
				              +   284*(INT(In[-17])+INT(In[17]))
				              +   340*(INT(In[-16])+INT(In[16]))
				              +   402*(INT(In[-15])+INT(In[15]))
				              +   470*(INT(In[-14])+INT(In[14]))
				              +   544*(INT(In[-13])+INT(In[13]))
				              +   623*(INT(In[-12])+INT(In[12]))
				              +   706*(INT(In[-11])+INT(In[11]))
				              +   792*(INT(In[-10])+INT(In[10]))
				              +   878*(INT(In[-9])+INT(In[9]))
				              +   962*(INT(In[-8])+INT(In[8]))
				              +   1044*(INT(In[-7])+INT(In[7]))
				              +   1120*(INT(In[-6])+INT(In[6]))
				              +   1189*(INT(In[-5])+INT(In[5]))
				              +   1249*(INT(In[-4])+INT(In[4]))
				              +   1297*(INT(In[-3])+INT(In[3]))
				              +   1333*(INT(In[-2])+INT(In[2]))
				              +   1354*(INT(In[-1])+INT(In[1]))
				              +   1362*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num94(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-38),-38,38,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num95 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num95(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-38])+INT(In[38]))
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   1*(INT(In[-36])+INT(In[36]))
				              +   2*(INT(In[-35])+INT(In[35]))
				              +   3*(INT(In[-34])+INT(In[34]))
				              +   4*(INT(In[-33])+INT(In[33]))
				              +   6*(INT(In[-32])+INT(In[32]))
				              +   8*(INT(In[-31])+INT(In[31]))
				              +   11*(INT(In[-30])+INT(In[30]))
				              +   15*(INT(In[-29])+INT(In[29]))
				              +   21*(INT(In[-28])+INT(In[28]))
				              +   28*(INT(In[-27])+INT(In[27]))
				              +   37*(INT(In[-26])+INT(In[26]))
				              +   49*(INT(In[-25])+INT(In[25]))
				              +   63*(INT(In[-24])+INT(In[24]))
				              +   81*(INT(In[-23])+INT(In[23]))
				              +   103*(INT(In[-22])+INT(In[22]))
				              +   129*(INT(In[-21])+INT(In[21]))
				              +   161*(INT(In[-20])+INT(In[20]))
				              +   198*(INT(In[-19])+INT(In[19]))
				              +   241*(INT(In[-18])+INT(In[18]))
				              +   290*(INT(In[-17])+INT(In[17]))
				              +   346*(INT(In[-16])+INT(In[16]))
				              +   408*(INT(In[-15])+INT(In[15]))
				              +   476*(INT(In[-14])+INT(In[14]))
				              +   549*(INT(In[-13])+INT(In[13]))
				              +   627*(INT(In[-12])+INT(In[12]))
				              +   709*(INT(In[-11])+INT(In[11]))
				              +   792*(INT(In[-10])+INT(In[10]))
				              +   876*(INT(In[-9])+INT(In[9]))
				              +   959*(INT(In[-8])+INT(In[8]))
				              +   1039*(INT(In[-7])+INT(In[7]))
				              +   1113*(INT(In[-6])+INT(In[6]))
				              +   1180*(INT(In[-5])+INT(In[5]))
				              +   1238*(INT(In[-4])+INT(In[4]))
				              +   1285*(INT(In[-3])+INT(In[3]))
				              +   1319*(INT(In[-2])+INT(In[2]))
				              +   1341*(INT(In[-1])+INT(In[1]))
				              +   1348*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num95(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-38),-38,38,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num96 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num96(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-38])+INT(In[38]))
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   2*(INT(In[-36])+INT(In[36]))
				              +   2*(INT(In[-35])+INT(In[35]))
				              +   3*(INT(In[-34])+INT(In[34]))
				              +   5*(INT(In[-33])+INT(In[33]))
				              +   6*(INT(In[-32])+INT(In[32]))
				              +   9*(INT(In[-31])+INT(In[31]))
				              +   12*(INT(In[-30])+INT(In[30]))
				              +   17*(INT(In[-29])+INT(In[29]))
				              +   23*(INT(In[-28])+INT(In[28]))
				              +   30*(INT(In[-27])+INT(In[27]))
				              +   40*(INT(In[-26])+INT(In[26]))
				              +   52*(INT(In[-25])+INT(In[25]))
				              +   67*(INT(In[-24])+INT(In[24]))
				              +   85*(INT(In[-23])+INT(In[23]))
				              +   107*(INT(In[-22])+INT(In[22]))
				              +   134*(INT(In[-21])+INT(In[21]))
				              +   166*(INT(In[-20])+INT(In[20]))
				              +   204*(INT(In[-19])+INT(In[19]))
				              +   247*(INT(In[-18])+INT(In[18]))
				              +   296*(INT(In[-17])+INT(In[17]))
				              +   352*(INT(In[-16])+INT(In[16]))
				              +   413*(INT(In[-15])+INT(In[15]))
				              +   481*(INT(In[-14])+INT(In[14]))
				              +   553*(INT(In[-13])+INT(In[13]))
				              +   630*(INT(In[-12])+INT(In[12]))
				              +   711*(INT(In[-11])+INT(In[11]))
				              +   793*(INT(In[-10])+INT(In[10]))
				              +   875*(INT(In[-9])+INT(In[9]))
				              +   956*(INT(In[-8])+INT(In[8]))
				              +   1034*(INT(In[-7])+INT(In[7]))
				              +   1106*(INT(In[-6])+INT(In[6]))
				              +   1171*(INT(In[-5])+INT(In[5]))
				              +   1227*(INT(In[-4])+INT(In[4]))
				              +   1273*(INT(In[-3])+INT(In[3]))
				              +   1306*(INT(In[-2])+INT(In[2]))
				              +   1327*(INT(In[-1])+INT(In[1]))
				              +   1334*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num96(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-38),-38,38,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num97 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num97(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-39])+INT(In[39]))
				              +   1*(INT(In[-38])+INT(In[38]))
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   2*(INT(In[-36])+INT(In[36]))
				              +   3*(INT(In[-35])+INT(In[35]))
				              +   4*(INT(In[-34])+INT(In[34]))
				              +   5*(INT(In[-33])+INT(In[33]))
				              +   7*(INT(In[-32])+INT(In[32]))
				              +   10*(INT(In[-31])+INT(In[31]))
				              +   13*(INT(In[-30])+INT(In[30]))
				              +   18*(INT(In[-29])+INT(In[29]))
				              +   24*(INT(In[-28])+INT(In[28]))
				              +   32*(INT(In[-27])+INT(In[27]))
				              +   42*(INT(In[-26])+INT(In[26]))
				              +   54*(INT(In[-25])+INT(In[25]))
				              +   70*(INT(In[-24])+INT(In[24]))
				              +   89*(INT(In[-23])+INT(In[23]))
				              +   112*(INT(In[-22])+INT(In[22]))
				              +   139*(INT(In[-21])+INT(In[21]))
				              +   172*(INT(In[-20])+INT(In[20]))
				              +   209*(INT(In[-19])+INT(In[19]))
				              +   253*(INT(In[-18])+INT(In[18]))
				              +   302*(INT(In[-17])+INT(In[17]))
				              +   358*(INT(In[-16])+INT(In[16]))
				              +   419*(INT(In[-15])+INT(In[15]))
				              +   486*(INT(In[-14])+INT(In[14]))
				              +   558*(INT(In[-13])+INT(In[13]))
				              +   633*(INT(In[-12])+INT(In[12]))
				              +   712*(INT(In[-11])+INT(In[11]))
				              +   793*(INT(In[-10])+INT(In[10]))
				              +   874*(INT(In[-9])+INT(In[9]))
				              +   953*(INT(In[-8])+INT(In[8]))
				              +   1028*(INT(In[-7])+INT(In[7]))
				              +   1099*(INT(In[-6])+INT(In[6]))
				              +   1162*(INT(In[-5])+INT(In[5]))
				              +   1217*(INT(In[-4])+INT(In[4]))
				              +   1261*(INT(In[-3])+INT(In[3]))
				              +   1294*(INT(In[-2])+INT(In[2]))
				              +   1314*(INT(In[-1])+INT(In[1]))
				              +   1320*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num97(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-39),-39,39,15){}
};

#define HAS_U_INT1_COMPILED_CONVOLUTIONS 

class cConvolSpec_U_INT1_Num98 : public cConvolSpec<U_INT1>
{
	public :
		bool IsCompiled() const { return true; }
		cConvolSpec<U_INT1> * duplicate() const { return new cConvolSpec_U_INT1_Num98(*this); }
		void Convol(U_INT1 *Out, const U_INT1 * In,int aK0,int aK1) const
		{
			In+=aK0;
			Out+=aK0;
			for (int aK=aK0; aK<aK1 ; aK++){
				*(Out++) =  (
				              +   1*(INT(In[-39])+INT(In[39]))
				              +   1*(INT(In[-38])+INT(In[38]))
				              +   1*(INT(In[-37])+INT(In[37]))
				              +   2*(INT(In[-36])+INT(In[36]))
				              +   3*(INT(In[-35])+INT(In[35]))
				              +   4*(INT(In[-34])+INT(In[34]))
				              +   6*(INT(In[-33])+INT(In[33]))
				              +   8*(INT(In[-32])+INT(In[32]))
				              +   11*(INT(In[-31])+INT(In[31]))
				              +   15*(INT(In[-30])+INT(In[30]))
				              +   20*(INT(In[-29])+INT(In[29]))
				              +   26*(INT(In[-28])+INT(In[28]))
				              +   34*(INT(In[-27])+INT(In[27]))
				              +   45*(INT(In[-26])+INT(In[26]))
				              +   57*(INT(In[-25])+INT(In[25]))
				              +   73*(INT(In[-24])+INT(In[24]))
				              +   93*(INT(In[-23])+INT(In[23]))
				              +   116*(INT(In[-22])+INT(In[22]))
				              +   144*(INT(In[-21])+INT(In[21]))
				              +   177*(INT(In[-20])+INT(In[20]))
				              +   215*(INT(In[-19])+INT(In[19]))
				              +   259*(INT(In[-18])+INT(In[18]))
				              +   308*(INT(In[-17])+INT(In[17]))
				              +   363*(INT(In[-16])+INT(In[16]))
				              +   424*(INT(In[-15])+INT(In[15]))
				              +   491*(INT(In[-14])+INT(In[14]))
				              +   562*(INT(In[-13])+INT(In[13]))
				              +   636*(INT(In[-12])+INT(In[12]))
				              +   714*(INT(In[-11])+INT(In[11]))
				              +   793*(INT(In[-10])+INT(In[10]))
				              +   872*(INT(In[-9])+INT(In[9]))
				              +   949*(INT(In[-8])+INT(In[8]))
				              +   1023*(INT(In[-7])+INT(In[7]))
				              +   1092*(INT(In[-6])+INT(In[6]))
				              +   1154*(INT(In[-5])+INT(In[5]))
				              +   1207*(INT(In[-4])+INT(In[4]))
				              +   1250*(INT(In[-3])+INT(In[3]))
				              +   1281*(INT(In[-2])+INT(In[2]))
				              +   1301*(INT(In[-1])+INT(In[1]))
				              +   1306*(In[0])
				            )>>15;
				In++;
			}
		}

		cConvolSpec_U_INT1_Num98(INT * aFilter):cConvolSpec<U_INT1>(aFilter-(-39),-39,39,15){}
};

template <> void ConvolutionHandler<U_INT1>::addCompiledKernels()
{
	{
		INT theCoeff[1] = {32768};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num0(theCoeff) );
	}
	{
		INT theCoeff[3] = {126,32516,126};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num1(theCoeff) );
	}
	{
		INT theCoeff[3] = {1323,30122,1323};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num2(theCoeff) );
	}
	{
		INT theCoeff[5] = {9,3488,25774,3488,9};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num3(theCoeff) );
	}
	{
		INT theCoeff[5] = {84,5424,21752,5424,84};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num4(theCoeff) );
	}
	{
		INT theCoeff[7] = {2,315,6731,18672,6731,315,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num5(theCoeff) );
	}
	{
		INT theCoeff[7] = {14,718,7481,16342,7481,718,14};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num6(theCoeff) );
	}
	{
		INT theCoeff[9] = {1,56,1230,7835,14524,7835,1230,56,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num7(theCoeff) );
	}
	{
		INT theCoeff[9] = {4,145,1769,7929,13074,7929,1769,145,4};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num8(theCoeff) );
	}
	{
		INT theCoeff[9] = {16,288,2276,7862,11884,7862,2276,288,16};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num9(theCoeff) );
	}
	{
		INT theCoeff[11] = {2,42,479,2716,7698,10894,7698,2716,479,42,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num10(theCoeff) );
	}
	{
		INT theCoeff[11] = {6,88,702,3079,7481,10056,7481,3079,702,88,6};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num11(theCoeff) );
	}
	{
		INT theCoeff[13] = {1,16,158,940,3366,7235,9336,7235,3366,940,158,16,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num12(theCoeff) );
	}
	{
		INT theCoeff[13] = {3,34,249,1179,3583,6978,8716,6978,3583,1179,249,34,3};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num13(theCoeff) );
	}
	{
		INT theCoeff[13] = {7,62,359,1409,3741,6721,8170,6721,3741,1409,359,62,7};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num14(theCoeff) );
	}
	{
		INT theCoeff[15] = {1,15,102,483,1621,3849,6468,7690,6468,3849,1621,483,102,15,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num15(theCoeff) );
	}
	{
		INT theCoeff[15] = {4,28,153,615,1811,3917,6224,7264,6224,3917,1811,615,153,28,4};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num16(theCoeff) );
	}
	{
		INT theCoeff[17] = {1,8,47,216,750,1978,3954,5990,6880,5990,3954,1978,750,216,47,8,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num17(theCoeff) );
	}
	{
		INT theCoeff[17] = {2,14,73,287,885,2122,3965,5768,6536,5768,3965,2122,885,287,73,14,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num18(theCoeff) );
	}
	{
		INT theCoeff[19] = {1,4,24,105,366,1015,2244,3955,5558,6224,5558,3955,2244,1015,366,105,24,4,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num19(theCoeff) );
	}
	{
		INT theCoeff[19] = {1,8,38,144,449,1138,2345,3931,5359,5942,5359,3931,2345,1138,449,144,38,8,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num20(theCoeff) );
	}
	{
		INT theCoeff[21] = {1,3,13,55,189,535,1253,2428,3894,5171,5684,5171,3894,2428,1253,535,189,55,13,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num21(theCoeff) );
	}
	{
		INT theCoeff[21] = {1,5,21,77,239,622,1358,2494,3849,4994,5448,4994,3849,2494,1358,622,239,77,21,5,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num22(theCoeff) );
	}
	{
		INT theCoeff[21] = {2,8,31,104,294,708,1454,2545,3797,4827,5228,4827,3797,2545,1454,708,294,104,31,8,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num23(theCoeff) );
	}
	{
		INT theCoeff[23] = {1,3,13,44,134,351,791,1540,2584,3740,4669,5028,4669,3740,2584,1540,791,351,134,44,13,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num24(theCoeff) );
	}
	{
		INT theCoeff[23] = {1,5,19,60,168,410,871,1616,2612,3680,4521,4842,4521,3680,2612,1616,871,410,168,60,19,5,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num25(theCoeff) );
	}
	{
		INT theCoeff[23] = {2,8,27,79,205,470,948,1683,2630,3618,4380,4668,4380,3618,2630,1683,948,470,205,79,27,8,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num26(theCoeff) );
	}
	{
		INT theCoeff[25] = {1,3,12,36,100,245,530,1020,1741,2640,3554,4248,4508,4248,3554,2640,1741,1020,530,245,100,36,12,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num27(theCoeff) );
	}
	{
		INT theCoeff[25] = {2,5,17,48,125,286,590,1087,1791,2643,3489,4122,4358,4122,3489,2643,1791,1087,590,286,125,48,17,5,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num28(theCoeff) );
	}
	{
		INT theCoeff[27] = {1,2,8,23,62,151,330,648,1148,1834,2640,3425,4003,4218,4003,3425,2640,1834,1148,648,330,151,62,23,8,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num29(theCoeff) );
	}
	{
		INT theCoeff[27] = {1,4,11,31,78,180,373,704,1205,1870,2633,3360,3891,4086,3891,3360,2633,1870,1205,704,373,180,78,31,11,4,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num30(theCoeff) );
	}
	{
		INT theCoeff[27] = {2,5,15,40,96,210,418,759,1257,1900,2621,3297,3784,3960,3784,3297,2621,1900,1257,759,418,210,96,40,15,5,2};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num31(theCoeff) );
	}
	{
		INT theCoeff[29] = {1,3,8,20,51,116,241,462,810,1304,1925,2605,3234,3682,3844,3682,3234,2605,1925,1304,810,462,241,116,51,20,8,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num32(theCoeff) );
	}
	{
		INT theCoeff[29] = {1,4,10,27,63,137,274,506,859,1346,1944,2587,3172,3586,3736,3586,3172,2587,1944,1346,859,506,274,137,63,27,10,4,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num33(theCoeff) );
	}
	{
		INT theCoeff[31] = {1,2,5,14,34,77,160,307,548,905,1384,1959,2566,3112,3494,3632,3494,3112,2566,1959,1384,905,548,307,160,77,34,14,5,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num34(theCoeff) );
	}
	{
		INT theCoeff[31] = {1,3,7,18,43,92,183,341,590,949,1418,1970,2543,3053,3406,3534,3406,3053,2543,1970,1418,949,590,341,183,92,43,18,7,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num35(theCoeff) );
	}
	{
		INT theCoeff[31] = {1,4,10,24,52,108,208,375,631,989,1448,1977,2519,2995,3323,3440,3323,2995,2519,1977,1448,989,631,375,208,108,52,24,10,4,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num36(theCoeff) );
	}
	{
		INT theCoeff[33] = {1,2,5,13,29,63,125,234,409,669,1026,1474,1981,2494,2939,3244,3352,3244,2939,2494,1981,1474,1026,669,409,234,125,63,29,13,5,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num37(theCoeff) );
	}
	{
		INT theCoeff[33] = {1,3,7,17,36,75,144,260,442,707,1061,1496,1982,2467,2884,3168,3268,3168,2884,2467,1982,1496,1061,707,442,260,144,75,36,17,7,3,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num38(theCoeff) );
	}
	{
		INT theCoeff[35] = {1,1,4,9,21,44,87,163,287,475,742,1093,1516,1981,2440,2831,3095,3188,3095,2831,2440,1981,1516,1093,742,475,287,163,87,44,21,9,4,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num39(theCoeff) );
	}
	{
		INT theCoeff[35] = {1,2,5,12,26,53,101,183,313,507,776,1122,1532,1978,2412,2779,3026,3112,3026,2779,2412,1978,1532,1122,776,507,313,183,101,53,26,12,5,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num40(theCoeff) );
	}
	{
		INT theCoeff[37] = {1,1,3,7,15,32,62,115,204,340,539,808,1148,1546,1972,2383,2729,2959,3040,2959,2729,2383,1972,1546,1148,808,539,340,204,115,62,32,15,7,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num41(theCoeff) );
	}
	{
		INT theCoeff[37] = {1,2,4,9,19,38,72,131,224,367,569,838,1173,1558,1965,2355,2679,2895,2970,2895,2679,2355,1965,1558,1173,838,569,367,224,131,72,38,19,9,4,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num42(theCoeff) );
	}
	{
		INT theCoeff[37] = {1,2,5,11,23,45,83,147,246,393,598,867,1194,1567,1957,2326,2632,2834,2906,2834,2632,2326,1957,1567,1194,867,598,393,246,147,83,45,23,11,5,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num43(theCoeff) );
	}
	{
		INT theCoeff[39] = {1,1,3,7,14,28,52,95,163,267,419,626,893,1214,1574,1947,2297,2586,2776,2842,2776,2586,2297,1947,1574,1214,893,626,419,267,163,95,52,28,14,7,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num44(theCoeff) );
	}
	{
		INT theCoeff[39] = {1,2,4,8,17,33,61,107,180,289,445,653,917,1231,1580,1936,2269,2541,2719,2782,2719,2541,2269,1936,1580,1231,917,653,445,289,180,107,61,33,17,8,4,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num45(theCoeff) );
	}
	{
		INT theCoeff[39] = {1,2,5,11,21,39,70,120,197,311,470,679,940,1247,1583,1925,2240,2497,2665,2722,2665,2497,2240,1925,1583,1247,940,679,470,311,197,120,70,39,21,11,5,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num46(theCoeff) );
	}
	{
		INT theCoeff[41] = {1,1,3,6,13,24,45,79,133,215,332,494,704,962,1261,1585,1912,2212,2455,2613,2668,2613,2455,2212,1912,1585,1261,962,704,494,332,215,133,79,45,24,13,6,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num47(theCoeff) );
	}
	{
		INT theCoeff[41] = {1,2,4,8,16,29,52,89,147,232,354,517,727,981,1273,1586,1899,2184,2413,2563,2614,2563,2413,2184,1899,1586,1273,981,727,517,354,232,147,89,52,29,16,8,4,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num48(theCoeff) );
	}
	{
		INT theCoeff[43] = {1,1,2,5,10,19,34,59,100,161,250,375,540,749,999,1283,1585,1885,2156,2374,2515,2562,2515,2374,2156,1885,1585,1283,999,749,540,375,250,161,100,59,34,19,10,5,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num49(theCoeff) );
	}
	{
		INT theCoeff[43] = {1,2,3,6,12,22,39,67,110,175,268,396,562,770,1016,1292,1584,1870,2129,2335,2468,2514,2468,2335,2129,1870,1584,1292,1016,770,562,396,268,175,110,67,39,22,12,6,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num50(theCoeff) );
	}
	{
		INT theCoeff[43] = {1,2,4,8,14,26,45,75,122,190,286,416,583,790,1031,1300,1581,1855,2101,2297,2423,2468,2423,2297,2101,1855,1581,1300,1031,790,583,416,286,190,122,75,45,26,14,8,4,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num51(theCoeff) );
	}
	{
		INT theCoeff[45] = {1,1,3,5,9,17,30,51,84,133,205,304,436,604,808,1045,1306,1577,1840,2075,2260,2380,2420,2380,2260,2075,1840,1577,1306,1045,808,604,436,304,205,133,84,51,30,17,9,5,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num52(theCoeff) );
	}
	{
		INT theCoeff[45] = {1,2,3,6,11,20,35,58,93,145,220,322,455,623,825,1057,1311,1572,1825,2048,2225,2338,2378,2338,2225,2048,1825,1572,1311,1057,825,623,455,322,220,145,93,58,35,20,11,6,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num53(theCoeff) );
	}
	{
		INT theCoeff[47] = {1,1,2,4,7,13,23,39,65,103,158,235,339,474,642,841,1069,1315,1567,1809,2022,2190,2298,2334,2298,2190,2022,1809,1567,1315,1069,841,642,474,339,235,158,103,65,39,23,13,7,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num54(theCoeff) );
	}
	{
		INT theCoeff[47] = {1,1,3,5,9,16,27,45,72,112,170,250,356,492,659,857,1079,1318,1561,1793,1997,2157,2258,2292,2258,2157,1997,1793,1561,1318,1079,857,659,492,356,250,170,112,72,45,27,16,9,5,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num55(theCoeff) );
	}
	{
		INT theCoeff[47] = {1,2,3,6,11,18,31,50,79,122,183,265,373,510,676,871,1088,1320,1554,1777,1972,2124,2221,2254,2221,2124,1972,1777,1554,1320,1088,871,676,510,373,265,183,122,79,50,31,18,11,6,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num56(theCoeff) );
	}
	{
		INT theCoeff[49] = {1,1,2,4,7,12,21,35,56,87,133,196,280,390,527,692,884,1096,1321,1547,1761,1947,2092,2184,2216,2184,2092,1947,1761,1547,1321,1096,884,692,527,390,280,196,133,87,56,35,21,12,7,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num57(theCoeff) );
	}
	{
		INT theCoeff[49] = {1,1,3,5,8,14,24,39,62,96,143,208,295,406,543,707,896,1103,1322,1540,1745,1923,2061,2149,2180,2149,2061,1923,1745,1540,1322,1103,896,707,543,406,295,208,143,96,62,39,24,14,8,5,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num58(theCoeff) );
	}
	{
		INT theCoeff[49] = {1,2,3,6,10,17,28,44,69,104,154,221,310,422,559,722,907,1109,1321,1532,1728,1899,2031,2114,2142,2114,2031,1899,1728,1532,1321,1109,907,722,559,422,310,221,154,104,69,44,28,17,10,6,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num59(theCoeff) );
	}
	{
		INT theCoeff[51] = {1,1,2,4,7,12,19,31,49,75,113,165,234,324,437,574,735,917,1115,1320,1523,1712,1876,2002,2081,2110,2081,2002,1876,1712,1523,1320,1115,917,735,574,437,324,234,165,113,75,49,31,19,12,7,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num60(theCoeff) );
	}
	{
		INT theCoeff[51] = {1,1,3,5,8,13,22,35,54,83,122,176,247,338,452,589,748,927,1119,1318,1514,1696,1853,1973,2049,2076,2049,1973,1853,1696,1514,1318,1119,927,748,589,452,338,247,176,122,83,54,35,22,13,8,5,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num61(theCoeff) );
	}
	{
		INT theCoeff[53] = {1,1,2,3,6,9,15,25,39,60,90,131,187,260,352,466,603,760,935,1123,1316,1505,1680,1830,1945,2018,2044,2018,1945,1830,1680,1505,1316,1123,935,760,603,466,352,260,187,131,90,60,39,25,15,9,6,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num62(theCoeff) );
	}
	{
		INT theCoeff[53] = {1,1,2,4,7,11,18,28,43,66,97,140,198,272,366,480,616,771,943,1126,1314,1496,1664,1808,1918,1988,2012,1988,1918,1808,1664,1496,1314,1126,943,771,616,480,366,272,198,140,97,66,43,28,18,11,7,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num63(theCoeff) );
	}
	{
		INT theCoeff[53] = {1,1,3,5,8,13,20,31,48,72,105,150,209,285,379,494,628,782,950,1129,1310,1487,1648,1786,1892,1958,1980,1958,1892,1786,1648,1487,1310,1129,950,782,628,494,379,285,209,150,105,72,48,31,20,13,8,5,3,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num64(theCoeff) );
	}
	{
		INT theCoeff[55] = {1,1,2,3,5,9,14,23,35,53,78,113,159,220,297,392,507,641,791,957,1130,1307,1477,1633,1765,1866,1930,1950,1930,1866,1765,1633,1477,1307,1130,957,791,641,507,392,297,220,159,113,78,53,35,23,14,9,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num65(theCoeff) );
	}
	{
		INT theCoeff[55] = {1,1,2,4,6,10,16,25,39,58,84,121,169,231,309,405,520,652,801,962,1132,1303,1467,1617,1744,1841,1902,1924,1902,1841,1744,1617,1467,1303,1132,962,801,652,520,405,309,231,169,121,84,58,39,25,16,10,6,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num66(theCoeff) );
	}
	{
		INT theCoeff[57] = {1,1,2,3,4,7,12,18,28,43,63,91,129,178,242,321,418,532,663,809,967,1132,1298,1457,1602,1724,1817,1875,1894,1875,1817,1724,1602,1457,1298,1132,967,809,663,532,418,321,242,178,129,91,63,43,28,18,12,7,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num67(theCoeff) );
	}
	{
		INT theCoeff[57] = {1,1,2,3,5,8,13,21,32,47,68,98,137,188,253,333,430,543,673,817,972,1133,1293,1447,1586,1704,1793,1849,1868,1849,1793,1704,1586,1447,1293,1133,972,817,673,543,430,333,253,188,137,98,68,47,32,21,13,8,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num68(theCoeff) );
	}
	{
		INT theCoeff[57] = {1,1,2,4,6,10,15,23,35,51,74,105,145,198,264,344,441,554,683,825,976,1133,1288,1437,1571,1684,1770,1823,1842,1823,1770,1684,1571,1437,1288,1133,976,825,683,554,441,344,264,198,145,105,74,51,35,23,15,10,6,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num69(theCoeff) );
	}
	{
		INT theCoeff[57] = {1,2,3,4,7,11,17,26,38,56,80,112,154,207,274,356,453,565,692,831,979,1132,1283,1427,1556,1665,1747,1798,1816,1798,1747,1665,1556,1427,1283,1132,979,831,692,565,453,356,274,207,154,112,80,56,38,26,17,11,7,4,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num70(theCoeff) );
	}
	{
		INT theCoeff[59] = {1,1,2,3,5,8,13,19,29,42,61,86,119,162,217,285,367,464,575,701,837,982,1131,1277,1416,1541,1646,1725,1774,1790,1774,1725,1646,1541,1416,1277,1131,982,837,701,575,464,367,285,217,162,119,86,61,42,29,19,13,8,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num71(theCoeff) );
	}
	{
		INT theCoeff[59] = {1,1,2,4,6,9,14,21,32,46,65,92,126,171,226,295,378,474,585,709,843,985,1129,1272,1406,1527,1627,1703,1751,1768,1751,1703,1627,1527,1406,1272,1129,985,843,709,585,474,378,295,226,171,126,92,65,46,32,21,14,9,6,4,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num72(theCoeff) );
	}
	{
		INT theCoeff[59] = {1,2,3,4,7,10,16,24,35,50,70,98,133,179,236,305,388,485,594,717,848,987,1128,1266,1396,1512,1609,1682,1728,1742,1728,1682,1609,1512,1396,1266,1128,987,848,717,594,485,388,305,236,179,133,98,70,50,35,24,16,10,7,4,3,2,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num73(theCoeff) );
	}
	{
		INT theCoeff[61] = {1,1,2,3,5,8,12,18,26,38,54,76,104,141,188,245,315,398,495,603,724,853,988,1125,1260,1385,1498,1591,1662,1705,1720,1705,1662,1591,1498,1385,1260,1125,988,853,724,603,495,398,315,245,188,141,104,76,54,38,26,18,12,8,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num74(theCoeff) );
	}
	{
		INT theCoeff[63] = {1,1,1,2,4,6,9,13,20,29,41,58,81,110,148,196,255,325,408,504,612,731,857,990,1123,1253,1375,1483,1574,1641,1684,1698,1684,1641,1574,1483,1375,1253,1123,990,857,731,612,504,408,325,255,196,148,110,81,58,41,29,20,13,9,6,4,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num75(theCoeff) );
	}
	{
		INT theCoeff[63] = {1,1,2,3,4,6,10,15,22,31,45,63,86,117,156,204,264,335,418,513,620,737,861,990,1120,1247,1365,1469,1557,1622,1662,1676,1662,1622,1557,1469,1365,1247,1120,990,861,737,620,513,418,335,264,204,156,117,86,63,45,31,22,15,10,6,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num76(theCoeff) );
	}
	{
		INT theCoeff[63] = {1,1,2,3,5,7,11,16,24,34,48,67,92,123,163,213,273,344,427,522,628,743,865,991,1118,1240,1354,1456,1540,1603,1642,1656,1642,1603,1540,1456,1354,1240,1118,991,865,743,628,522,427,344,273,213,163,123,92,67,48,34,24,16,11,7,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num77(theCoeff) );
	}
	{
		INT theCoeff[65] = {1,1,2,2,4,6,8,12,18,26,37,52,72,97,130,171,221,282,353,436,531,635,748,868,991,1114,1234,1344,1442,1523,1584,1622,1634,1622,1584,1523,1442,1344,1234,1114,991,868,748,635,531,436,353,282,221,171,130,97,72,52,37,26,18,12,8,6,4,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num78(theCoeff) );
	}
	{
		INT theCoeff[65] = {1,1,2,3,4,6,9,14,20,29,40,56,77,103,137,178,229,290,362,445,539,642,753,871,991,1111,1227,1334,1429,1507,1565,1602,1614,1602,1565,1507,1429,1334,1227,1111,991,871,753,642,539,445,362,290,229,178,137,103,77,56,40,29,20,14,9,6,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num79(theCoeff) );
	}
	{
		INT theCoeff[67] = {1,1,1,2,3,5,7,10,15,22,31,44,60,81,109,143,186,238,299,371,454,546,648,758,873,991,1107,1220,1324,1415,1491,1548,1583,1594,1583,1548,1491,1415,1324,1220,1107,991,873,758,648,546,454,371,299,238,186,143,109,81,60,44,31,22,15,10,7,5,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num80(theCoeff) );
	}
	{
		INT theCoeff[67] = {1,1,1,2,4,5,8,12,17,24,34,47,64,86,115,150,193,246,308,380,462,554,654,762,875,990,1104,1213,1314,1402,1475,1530,1564,1574,1564,1530,1475,1402,1314,1213,1104,990,875,762,654,554,462,380,308,246,193,150,115,86,64,47,34,24,17,12,8,5,4,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num81(theCoeff) );
	}
	{
		INT theCoeff[67] = {1,1,2,3,4,6,9,13,19,26,37,50,68,91,120,157,201,254,316,388,470,561,660,766,877,989,1100,1206,1304,1389,1460,1513,1545,1556,1545,1513,1460,1389,1304,1206,1100,989,877,766,660,561,470,388,316,254,201,157,120,91,68,50,37,26,19,13,9,6,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num82(theCoeff) );
	}
	{
		INT theCoeff[67] = {1,1,2,3,5,7,10,14,20,29,39,54,73,97,126,163,208,261,324,396,478,568,666,770,878,988,1096,1199,1294,1377,1445,1496,1527,1538,1527,1496,1445,1377,1294,1199,1096,988,878,770,666,568,478,396,324,261,208,163,126,97,73,54,39,29,20,14,10,7,5,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num83(theCoeff) );
	}
	{
		INT theCoeff[69] = {1,1,2,2,3,5,8,11,16,22,31,43,58,77,102,132,170,215,269,332,404,485,574,671,773,879,986,1092,1192,1284,1364,1430,1480,1510,1520,1510,1480,1430,1364,1284,1192,1092,986,879,773,671,574,485,404,332,269,215,170,132,102,77,58,43,31,22,16,11,8,5,3,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num84(theCoeff) );
	}
	{
		INT theCoeff[69] = {1,1,2,3,4,6,8,12,17,24,33,46,61,82,107,138,177,223,277,340,412,492,580,676,776,880,985,1087,1185,1274,1352,1416,1463,1493,1502,1493,1463,1416,1352,1274,1185,1087,985,880,776,676,580,492,412,340,277,223,177,138,107,82,61,46,33,24,17,12,8,6,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num85(theCoeff) );
	}
	{
		INT theCoeff[71] = {1,1,1,2,3,4,7,9,13,19,26,36,49,65,86,112,144,183,230,285,348,419,499,586,680,779,881,983,1083,1177,1264,1340,1402,1448,1476,1486,1476,1448,1402,1340,1264,1177,1083,983,881,779,680,586,499,419,348,285,230,183,144,112,86,65,49,36,26,19,13,9,7,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num86(theCoeff) );
	}
	{
		INT theCoeff[71] = {1,1,2,2,3,5,7,10,15,21,28,39,52,69,91,118,150,190,237,292,355,426,506,592,684,781,881,981,1078,1170,1254,1328,1388,1432,1460,1470,1460,1432,1388,1328,1254,1170,1078,981,881,781,684,592,506,426,355,292,237,190,150,118,91,69,52,39,28,21,15,10,7,5,3,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num87(theCoeff) );
	}
	{
		INT theCoeff[71] = {1,1,2,3,4,6,8,11,16,22,31,42,55,73,95,123,156,197,244,299,362,433,512,597,688,784,881,979,1073,1163,1245,1316,1374,1417,1444,1454,1444,1417,1374,1316,1245,1163,1073,979,881,784,688,597,512,433,362,299,244,197,156,123,95,73,55,42,31,22,16,11,8,6,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num88(theCoeff) );
	}
	{
		INT theCoeff[73] = {1,1,1,2,3,4,6,9,13,18,24,33,44,59,77,100,128,163,203,251,306,369,440,518,602,692,786,881,976,1069,1156,1235,1304,1361,1402,1428,1438,1428,1402,1361,1304,1235,1156,1069,976,881,786,692,602,518,440,369,306,251,203,163,128,100,77,59,44,33,24,18,13,9,6,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num89(theCoeff) );
	}
	{
		INT theCoeff[73] = {1,1,2,2,3,5,7,10,14,19,26,35,47,62,81,105,134,168,210,258,313,376,446,524,607,695,787,881,974,1064,1149,1226,1293,1347,1388,1413,1422,1413,1388,1347,1293,1226,1149,1064,974,881,787,695,607,524,446,376,313,258,210,168,134,105,81,62,47,35,26,19,14,10,7,5,3,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num90(theCoeff) );
	}
	{
		INT theCoeff[73] = {1,1,2,3,4,5,8,11,15,21,28,38,50,66,86,110,139,174,216,264,320,383,453,529,611,698,789,880,971,1059,1142,1217,1281,1334,1374,1398,1406,1398,1374,1334,1281,1217,1142,1059,971,880,789,698,611,529,453,383,320,264,216,174,139,110,86,66,50,38,28,21,15,11,8,5,4,3,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num91(theCoeff) );
	}
	{
		INT theCoeff[75] = {1,1,1,2,3,4,6,9,12,16,22,30,41,53,70,90,115,145,180,222,271,327,389,459,535,616,701,790,879,968,1054,1134,1207,1270,1322,1360,1383,1392,1383,1360,1322,1270,1207,1134,1054,968,879,790,701,616,535,459,389,327,271,222,180,145,115,90,70,53,41,30,22,16,12,9,6,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num92(theCoeff) );
	}
	{
		INT theCoeff[75] = {1,1,2,2,3,5,7,9,13,18,24,32,43,57,73,94,120,150,186,229,278,333,396,465,540,620,704,791,879,965,1049,1127,1198,1259,1309,1346,1368,1376,1368,1346,1309,1259,1198,1127,1049,965,879,791,704,620,540,465,396,333,278,229,186,150,120,94,73,57,43,32,24,18,13,9,7,5,3,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num93(theCoeff) );
	}
	{
		INT theCoeff[77] = {1,1,1,2,3,4,5,7,10,14,19,26,35,46,60,77,99,124,155,192,235,284,340,402,470,544,623,706,792,878,962,1044,1120,1189,1249,1297,1333,1354,1362,1354,1333,1297,1249,1189,1120,1044,962,878,792,706,623,544,470,402,340,284,235,192,155,124,99,77,60,46,35,26,19,14,10,7,5,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num94(theCoeff) );
	}
	{
		INT theCoeff[77] = {1,1,1,2,3,4,6,8,11,15,21,28,37,49,63,81,103,129,161,198,241,290,346,408,476,549,627,709,792,876,959,1039,1113,1180,1238,1285,1319,1341,1348,1341,1319,1285,1238,1180,1113,1039,959,876,792,709,627,549,476,408,346,290,241,198,161,129,103,81,63,49,37,28,21,15,11,8,6,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num95(theCoeff) );
	}
	{
		INT theCoeff[77] = {1,1,2,2,3,5,6,9,12,17,23,30,40,52,67,85,107,134,166,204,247,296,352,413,481,553,630,711,793,875,956,1034,1106,1171,1227,1273,1306,1327,1334,1327,1306,1273,1227,1171,1106,1034,956,875,793,711,630,553,481,413,352,296,247,204,166,134,107,85,67,52,40,30,23,17,12,9,6,5,3,2,2,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num96(theCoeff) );
	}
	{
		INT theCoeff[79] = {1,1,1,2,3,4,5,7,10,13,18,24,32,42,54,70,89,112,139,172,209,253,302,358,419,486,558,633,712,793,874,953,1028,1099,1162,1217,1261,1294,1314,1320,1314,1294,1261,1217,1162,1099,1028,953,874,793,712,633,558,486,419,358,302,253,209,172,139,112,89,70,54,42,32,24,18,13,10,7,5,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num97(theCoeff) );
	}
	{
		INT theCoeff[79] = {1,1,1,2,3,4,6,8,11,15,20,26,34,45,57,73,93,116,144,177,215,259,308,363,424,491,562,636,714,793,872,949,1023,1092,1154,1207,1250,1281,1301,1306,1301,1281,1250,1207,1154,1092,1023,949,872,793,714,636,562,491,424,363,308,259,215,177,144,116,93,73,57,45,34,26,20,15,11,8,6,4,3,2,1,1,1};
		mConvolutions.push_back( new cConvolSpec_U_INT1_Num98(theCoeff) );
	}
}
