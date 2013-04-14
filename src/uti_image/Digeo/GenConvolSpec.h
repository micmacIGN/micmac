#include "Digeo.h"

/* Sigma 1.600000  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num0 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
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

      cConvolSpec_U_INT2_Num0(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.226273  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num1 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
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

      cConvolSpec_U_INT2_Num1(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.545008  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num2 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
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

      cConvolSpec_U_INT2_Num2(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.946588  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num3 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
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

      cConvolSpec_U_INT2_Num3(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.452547  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num4 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
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

      cConvolSpec_U_INT2_Num4(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.090016  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num5 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   2*(In[-12]+In[12])
                              +   8*(In[-11]+In[11])
                              +   23*(In[-10]+In[10])
                              +   63*(In[-9]+In[9])
                              +   152*(In[-8]+In[8])
                              +   331*(In[-7]+In[7])
                              +   650*(In[-6]+In[6])
                              +   1151*(In[-5]+In[5])
                              +   1836*(In[-4]+In[4])
                              +   2640*(In[-3]+In[3])
                              +   3423*(In[-2]+In[2])
                              +   3999*(In[-1]+In[1])
                              +   4212*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num5(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 1.600000  ModeIncrem 1 */
class cConvolSpec_REAL4_Num0 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000269*(In[-6]+In[6])
                              +   0.002164*(In[-5]+In[5])
                              +   0.011894*(In[-4]+In[4])
                              +   0.044730*(In[-3]+In[3])
                              +   0.115169*(In[-2]+In[2])
                              +   0.203094*(In[-1]+In[1])
                              +   0.245360*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num0(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.226273  ModeIncrem 1 */
class cConvolSpec_REAL4_Num1 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000118*(In[-5]+In[5])
                              +   0.002035*(In[-4]+In[4])
                              +   0.018578*(In[-3]+In[3])
                              +   0.089876*(In[-2]+In[2])
                              +   0.231116*(In[-1]+In[1])
                              +   0.316556*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num1(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.545008  ModeIncrem 1 */
class cConvolSpec_REAL4_Num2 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000173*(In[-6]+In[6])
                              +   0.001606*(In[-5]+In[5])
                              +   0.009951*(In[-4]+In[4])
                              +   0.041071*(In[-3]+In[3])
                              +   0.112988*(In[-2]+In[2])
                              +   0.207315*(In[-1]+In[1])
                              +   0.253793*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num2(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.946588  ModeIncrem 1 */
class cConvolSpec_REAL4_Num3 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000052*(In[-8]+In[8])
                              +   0.000362*(In[-7]+In[7])
                              +   0.001940*(In[-6]+In[6])
                              +   0.008034*(In[-5]+In[5])
                              +   0.025689*(In[-4]+In[4])
                              +   0.063430*(In[-3]+In[3])
                              +   0.120961*(In[-2]+In[2])
                              +   0.178171*(In[-1]+In[1])
                              +   0.202720*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num3(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.452547  ModeIncrem 1 */
class cConvolSpec_REAL4_Num4 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000044*(In[-10]+In[10])
                              +   0.000211*(In[-9]+In[9])
                              +   0.000849*(In[-8]+In[8])
                              +   0.002907*(In[-7]+In[7])
                              +   0.008441*(In[-6]+In[6])
                              +   0.020803*(In[-5]+In[5])
                              +   0.043511*(In[-4]+In[4])
                              +   0.077242*(In[-3]+In[3])
                              +   0.116382*(In[-2]+In[2])
                              +   0.148834*(In[-1]+In[1])
                              +   0.161550*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num4(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.090016  ModeIncrem 1 */
class cConvolSpec_REAL4_Num5 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000073*(In[-12]+In[12])
                              +   0.000240*(In[-11]+In[11])
                              +   0.000715*(In[-10]+In[10])
                              +   0.001918*(In[-9]+In[9])
                              +   0.004636*(In[-8]+In[8])
                              +   0.010100*(In[-7]+In[7])
                              +   0.019836*(In[-6]+In[6])
                              +   0.035112*(In[-5]+In[5])
                              +   0.056022*(In[-4]+In[4])
                              +   0.080571*(In[-3]+In[3])
                              +   0.104449*(In[-2]+In[2])
                              +   0.122051*(In[-1]+In[1])
                              +   0.128554*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num5(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 1.029751  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num6 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   11*(In[-4]+In[4])
                              +   238*(In[-3]+In[3])
                              +   2130*(In[-2]+In[2])
                              +   7898*(In[-1]+In[1])
                              +   12214*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num6(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.224587  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num7 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   4*(In[-5]+In[5])
                              +   66*(In[-4]+In[4])
                              +   605*(In[-3]+In[3])
                              +   2939*(In[-2]+In[2])
                              +   7577*(In[-1]+In[1])
                              +   10386*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num7(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.456288  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num8 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   3*(In[-6]+In[6])
                              +   30*(In[-5]+In[5])
                              +   233*(In[-4]+In[4])
                              +   1143*(In[-3]+In[3])
                              +   3555*(In[-2]+In[2])
                              +   7018*(In[-1]+In[1])
                              +   8804*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num8(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.731828  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num9 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   3*(In[-7]+In[7])
                              +   22*(In[-6]+In[6])
                              +   129*(In[-5]+In[5])
                              +   555*(In[-4]+In[4])
                              +   1730*(In[-3]+In[3])
                              +   3892*(In[-2]+In[2])
                              +   6331*(In[-1]+In[1])
                              +   7444*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num9(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.059502  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num10 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   4*(In[-8]+In[8])
                              +   22*(In[-7]+In[7])
                              +   98*(In[-6]+In[6])
                              +   349*(In[-5]+In[5])
                              +   988*(In[-4]+In[4])
                              +   2221*(In[-3]+In[3])
                              +   3959*(In[-2]+In[2])
                              +   5600*(In[-1]+In[1])
                              +   6286*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num10(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.449174  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num11 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-10]+In[10])
                              +   7*(In[-9]+In[9])
                              +   27*(In[-8]+In[8])
                              +   94*(In[-7]+In[7])
                              +   275*(In[-6]+In[6])
                              +   679*(In[-5]+In[5])
                              +   1423*(In[-4]+In[4])
                              +   2529*(In[-3]+In[3])
                              +   3815*(In[-2]+In[2])
                              +   4883*(In[-1]+In[1])
                              +   5302*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num11(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 1.029751  ModeIncrem 1 */
class cConvolSpec_REAL4_Num6 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000332*(In[-4]+In[4])
                              +   0.007254*(In[-3]+In[3])
                              +   0.064996*(In[-2]+In[2])
                              +   0.241042*(In[-1]+In[1])
                              +   0.372752*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num6(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.224587  ModeIncrem 1 */
class cConvolSpec_REAL4_Num7 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000115*(In[-5]+In[5])
                              +   0.002011*(In[-4]+In[4])
                              +   0.018464*(In[-3]+In[3])
                              +   0.089698*(In[-2]+In[2])
                              +   0.231227*(In[-1]+In[1])
                              +   0.316968*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num7(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.456288  ModeIncrem 1 */
class cConvolSpec_REAL4_Num8 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000075*(In[-6]+In[6])
                              +   0.000921*(In[-5]+In[5])
                              +   0.007120*(In[-4]+In[4])
                              +   0.034891*(In[-3]+In[3])
                              +   0.108481*(In[-2]+In[2])
                              +   0.214177*(In[-1]+In[1])
                              +   0.268670*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num8(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.731828  ModeIncrem 1 */
class cConvolSpec_REAL4_Num9 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000080*(In[-7]+In[7])
                              +   0.000660*(In[-6]+In[6])
                              +   0.003935*(In[-5]+In[5])
                              +   0.016956*(In[-4]+In[4])
                              +   0.052788*(In[-3]+In[3])
                              +   0.118777*(In[-2]+In[2])
                              +   0.193200*(In[-1]+In[1])
                              +   0.227209*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num9(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.059502  ModeIncrem 1 */
class cConvolSpec_REAL4_Num10 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000117*(In[-8]+In[8])
                              +   0.000664*(In[-7]+In[7])
                              +   0.002987*(In[-6]+In[6])
                              +   0.010658*(In[-5]+In[5])
                              +   0.030173*(In[-4]+In[4])
                              +   0.067778*(In[-3]+In[3])
                              +   0.120814*(In[-2]+In[2])
                              +   0.170893*(In[-1]+In[1])
                              +   0.191833*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num10(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.449174  ModeIncrem 1 */
class cConvolSpec_REAL4_Num11 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000043*(In[-10]+In[10])
                              +   0.000207*(In[-9]+In[9])
                              +   0.000839*(In[-8]+In[8])
                              +   0.002879*(In[-7]+In[7])
                              +   0.008385*(In[-6]+In[6])
                              +   0.020715*(In[-5]+In[5])
                              +   0.043415*(In[-4]+In[4])
                              +   0.077192*(In[-3]+In[3])
                              +   0.116436*(In[-2]+In[2])
                              +   0.149003*(In[-1]+In[1])
                              +   0.161770*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num11(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 0.904400  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num12 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   2*(In[-4]+In[4])
                              +   92*(In[-3]+In[3])
                              +   1498*(In[-2]+In[2])
                              +   7916*(In[-1]+In[1])
                              +   13752*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num12(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.038883  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num13 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   12*(In[-4]+In[4])
                              +   251*(In[-3]+In[3])
                              +   2173*(In[-2]+In[2])
                              +   7890*(In[-1]+In[1])
                              +   12116*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num13(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.193364  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num14 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   3*(In[-5]+In[5])
                              +   52*(In[-4]+In[4])
                              +   538*(In[-3]+In[3])
                              +   2827*(In[-2]+In[2])
                              +   7643*(In[-1]+In[1])
                              +   10642*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num14(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.370815  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num15 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-6]+In[6])
                              +   16*(In[-5]+In[5])
                              +   158*(In[-4]+In[4])
                              +   942*(In[-3]+In[3])
                              +   3369*(In[-2]+In[2])
                              +   7233*(In[-1]+In[1])
                              +   9330*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.574653  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num16 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   7*(In[-6]+In[6])
                              +   62*(In[-5]+In[5])
                              +   360*(In[-4]+In[4])
                              +   1411*(In[-3]+In[3])
                              +   3743*(In[-2]+In[2])
                              +   6719*(In[-1]+In[1])
                              +   8164*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num16(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.808801  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num17 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   5*(In[-7]+In[7])
                              +   33*(In[-6]+In[6])
                              +   172*(In[-5]+In[5])
                              +   658*(In[-4]+In[4])
                              +   1867*(In[-3]+In[3])
                              +   3932*(In[-2]+In[2])
                              +   6149*(In[-1]+In[1])
                              +   7136*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num17(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.077767  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num18 : public cConvolSpec<U_INT2>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   4*(In[-8]+In[8])
                              +   24*(In[-7]+In[7])
                              +   104*(In[-6]+In[6])
                              +   364*(In[-5]+In[5])
                              +   1012*(In[-4]+In[4])
                              +   2241*(In[-3]+In[3])
                              +   3956*(In[-2]+In[2])
                              +   5563*(In[-1]+In[1])
                              +   6232*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num18(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 0.904400  ModeIncrem 1 */
class cConvolSpec_REAL4_Num12 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000054*(In[-4]+In[4])
                              +   0.002795*(In[-3]+In[3])
                              +   0.045733*(In[-2]+In[2])
                              +   0.241577*(In[-1]+In[1])
                              +   0.419680*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num12(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.038883  ModeIncrem 1 */
class cConvolSpec_REAL4_Num13 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000369*(In[-4]+In[4])
                              +   0.007673*(In[-3]+In[3])
                              +   0.066323*(In[-2]+In[2])
                              +   0.240774*(In[-1]+In[1])
                              +   0.369722*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num13(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.193364  ModeIncrem 1 */
class cConvolSpec_REAL4_Num14 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000079*(In[-5]+In[5])
                              +   0.001597*(In[-4]+In[4])
                              +   0.016405*(In[-3]+In[3])
                              +   0.086288*(In[-2]+In[2])
                              +   0.233234*(In[-1]+In[1])
                              +   0.324794*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num14(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.370815  ModeIncrem 1 */
class cConvolSpec_REAL4_Num15 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000029*(In[-6]+In[6])
                              +   0.000484*(In[-5]+In[5])
                              +   0.004821*(In[-4]+In[4])
                              +   0.028755*(In[-3]+In[3])
                              +   0.102824*(In[-2]+In[2])
                              +   0.220731*(In[-1]+In[1])
                              +   0.284714*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.574653  ModeIncrem 1 */
class cConvolSpec_REAL4_Num16 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000221*(In[-6]+In[6])
                              +   0.001894*(In[-5]+In[5])
                              +   0.010983*(In[-4]+In[4])
                              +   0.043063*(In[-3]+In[3])
                              +   0.114218*(In[-2]+In[2])
                              +   0.205034*(In[-1]+In[1])
                              +   0.249177*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num16(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.808801  ModeIncrem 1 */
class cConvolSpec_REAL4_Num17 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000146*(In[-7]+In[7])
                              +   0.001017*(In[-6]+In[6])
                              +   0.005245*(In[-5]+In[5])
                              +   0.020069*(In[-4]+In[4])
                              +   0.056969*(In[-3]+In[3])
                              +   0.120011*(In[-2]+In[2])
                              +   0.187647*(In[-1]+In[1])
                              +   0.217793*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num17(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.077767  ModeIncrem 1 */
class cConvolSpec_REAL4_Num18 : public cConvolSpec<REAL4>
{
   public :
      bool IsCompiled() const {return true;}
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000132*(In[-8]+In[8])
                              +   0.000725*(In[-7]+In[7])
                              +   0.003180*(In[-6]+In[6])
                              +   0.011104*(In[-5]+In[5])
                              +   0.030878*(In[-4]+In[4])
                              +   0.068406*(In[-3]+In[3])
                              +   0.120728*(In[-2]+In[2])
                              +   0.169757*(In[-1]+In[1])
                              +   0.190180*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num18(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

