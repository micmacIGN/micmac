#include "Digeo.h"

/* Sigma 1.226273  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num0(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.545008  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num1(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.946588  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num2(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.452547  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num3(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.090016  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num4(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 0.766421  ModeIncrem 1 */
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
                              +   0.000550*(In[-3]+In[3])
                              +   0.024594*(In[-2]+In[2])
                              +   0.231898*(In[-1]+In[1])
                              +   0.485916*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num5(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.965630  ModeIncrem 1 */
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
                              +   0.000143*(In[-4]+In[4])
                              +   0.004665*(In[-3]+In[3])
                              +   0.055337*(In[-2]+In[2])
                              +   0.242136*(In[-1]+In[1])
                              +   0.395438*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num6(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.216617  ModeIncrem 1 */
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
                              +   0.000105*(In[-5]+In[5])
                              +   0.001899*(In[-4]+In[4])
                              +   0.017932*(In[-3]+In[3])
                              +   0.088848*(In[-2]+In[2])
                              +   0.231750*(In[-1]+In[1])
                              +   0.318931*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num7(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.532842  ModeIncrem 1 */
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
                              +   0.000155*(In[-6]+In[6])
                              +   0.001497*(In[-5]+In[5])
                              +   0.009539*(In[-4]+In[4])
                              +   0.040241*(In[-3]+In[3])
                              +   0.112445*(In[-2]+In[2])
                              +   0.208254*(In[-1]+In[1])
                              +   0.255736*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num8(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.931260  ModeIncrem 1 */
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
                              +   0.000046*(In[-8]+In[8])
                              +   0.000330*(In[-7]+In[7])
                              +   0.001818*(In[-6]+In[6])
                              +   0.007699*(In[-5]+In[5])
                              +   0.025068*(In[-4]+In[4])
                              +   0.062776*(In[-3]+In[3])
                              +   0.120923*(In[-2]+In[2])
                              +   0.179193*(In[-1]+In[1])
                              +   0.204294*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num9(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 1.226273  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num0(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.545008  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num1(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.946588  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num2(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.452547  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num3(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.090016  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num4(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 0.766421  ModeIncrem 1 */
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
                              +   18*(In[-3]+In[3])
                              +   806*(In[-2]+In[2])
                              +   7599*(In[-1]+In[1])
                              +   15922*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num5(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.965630  ModeIncrem 1 */
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
                              +   5*(In[-4]+In[4])
                              +   153*(In[-3]+In[3])
                              +   1813*(In[-2]+In[2])
                              +   7934*(In[-1]+In[1])
                              +   12958*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num6(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.216617  ModeIncrem 1 */
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
                              +   3*(In[-5]+In[5])
                              +   62*(In[-4]+In[4])
                              +   588*(In[-3]+In[3])
                              +   2911*(In[-2]+In[2])
                              +   7594*(In[-1]+In[1])
                              +   10452*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num7(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.532842  ModeIncrem 1 */
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
                              +   5*(In[-6]+In[6])
                              +   49*(In[-5]+In[5])
                              +   312*(In[-4]+In[4])
                              +   1319*(In[-3]+In[3])
                              +   3685*(In[-2]+In[2])
                              +   6824*(In[-1]+In[1])
                              +   8380*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num8(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.931260  ModeIncrem 1 */
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
                              +   2*(In[-8]+In[8])
                              +   11*(In[-7]+In[7])
                              +   60*(In[-6]+In[6])
                              +   252*(In[-5]+In[5])
                              +   821*(In[-4]+In[4])
                              +   2057*(In[-3]+In[3])
                              +   3962*(In[-2]+In[2])
                              +   5872*(In[-1]+In[1])
                              +   6694*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num9(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 1.029751  ModeIncrem 1 */
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
                              +   11*(In[-4]+In[4])
                              +   238*(In[-3]+In[3])
                              +   2130*(In[-2]+In[2])
                              +   7898*(In[-1]+In[1])
                              +   12214*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num10(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.224587  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num11(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.456288  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num12(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.731828  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num13(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.059502  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num14(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.449174  ModeIncrem 1 */
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

      cConvolSpec_U_INT2_Num15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 0.643594  ModeIncrem 1 */
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
                              +   2*(In[-3]+In[3])
                              +   322*(In[-2]+In[2])
                              +   6838*(In[-1]+In[1])
                              +   18444*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num16(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.765367  ModeIncrem 1 */
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
                              +   18*(In[-3]+In[3])
                              +   801*(In[-2]+In[2])
                              +   7595*(In[-1]+In[1])
                              +   15940*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num17(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.910180  ModeIncrem 1 */
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
                              +   2*(In[-4]+In[4])
                              +   97*(In[-3]+In[3])
                              +   1529*(In[-2]+In[2])
                              +   7920*(In[-1]+In[1])
                              +   13672*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num18(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.082392  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num19 : public cConvolSpec<U_INT2>
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
                              +   1*(In[-5]+In[5])
                              +   19*(In[-4]+In[4])
                              +   322*(In[-3]+In[3])
                              +   2374*(In[-2]+In[2])
                              +   7837*(In[-1]+In[1])
                              +   11662*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num19(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.287189  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num20 : public cConvolSpec<U_INT2>
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
                              +   7*(In[-5]+In[5])
                              +   100*(In[-4]+In[4])
                              +   746*(In[-3]+In[3])
                              +   3142*(In[-2]+In[2])
                              +   7435*(In[-1]+In[1])
                              +   9908*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num20(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.530734  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num21 : public cConvolSpec<U_INT2>
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
                              +   5*(In[-6]+In[6])
                              +   48*(In[-5]+In[5])
                              +   310*(In[-4]+In[4])
                              +   1314*(In[-3]+In[3])
                              +   3682*(In[-2]+In[2])
                              +   6829*(In[-1]+In[1])
                              +   8392*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num21(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 0.904400  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num22 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num22(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.038883  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num23 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num23(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.193364  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num24 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num24(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.370815  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num25 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num25(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.574653  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num26 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num26(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.808801  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num27 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num27(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.077767  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num28 : public cConvolSpec<U_INT2>
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

      cConvolSpec_U_INT2_Num28(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 0.565250  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num29 : public cConvolSpec<U_INT2>
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
                              +   130*(In[-2]+In[2])
                              +   6034*(In[-1]+In[1])
                              +   20440*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num29(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-2),-2,2,15,false)       {
      }
};

/* Sigma 0.649302  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num30 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-3]+In[3])
                              +   340*(In[-2]+In[2])
                              +   6886*(In[-1]+In[1])
                              +   18312*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num30(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.745852  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num31 : public cConvolSpec<U_INT2>
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
                              +   13*(In[-3]+In[3])
                              +   712*(In[-2]+In[2])
                              +   7508*(In[-1]+In[1])
                              +   16302*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num31(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.856759  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num32 : public cConvolSpec<U_INT2>
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
                              +   1*(In[-4]+In[4])
                              +   57*(In[-3]+In[3])
                              +   1252*(In[-2]+In[2])
                              +   7856*(In[-1]+In[1])
                              +   14436*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num32(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 0.984158  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num33 : public cConvolSpec<U_INT2>
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
                              +   6*(In[-4]+In[4])
                              +   175*(In[-3]+In[3])
                              +   1907*(In[-2]+In[2])
                              +   7929*(In[-1]+In[1])
                              +   12734*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num33(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.130501  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num34 : public cConvolSpec<U_INT2>
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
                              +   1*(In[-5]+In[5])
                              +   31*(In[-4]+In[4])
                              +   410*(In[-3]+In[3])
                              +   2581*(In[-2]+In[2])
                              +   7762*(In[-1]+In[1])
                              +   11198*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num34(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.298604  ModeIncrem 1 */
class cConvolSpec_U_INT2_Num35 : public cConvolSpec<U_INT2>
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
                              +   8*(In[-5]+In[5])
                              +   106*(In[-4]+In[4])
                              +   773*(In[-3]+In[3])
                              +   3176*(In[-2]+In[2])
                              +   7409*(In[-1]+In[1])
                              +   9824*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num35(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.029751  ModeIncrem 1 */
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
                              +   0.000332*(In[-4]+In[4])
                              +   0.007254*(In[-3]+In[3])
                              +   0.064996*(In[-2]+In[2])
                              +   0.241042*(In[-1]+In[1])
                              +   0.372752*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num10(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.224587  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num11(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.456288  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num12(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.731828  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num13(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.059502  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num14(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.449174  ModeIncrem 1 */
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

      cConvolSpec_REAL4_Num15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 0.643594  ModeIncrem 1 */
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
                              +   0.000051*(In[-3]+In[3])
                              +   0.009821*(In[-2]+In[2])
                              +   0.208689*(In[-1]+In[1])
                              +   0.562879*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num16(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.765367  ModeIncrem 1 */
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
                              +   0.000541*(In[-3]+In[3])
                              +   0.024445*(In[-2]+In[2])
                              +   0.231766*(In[-1]+In[1])
                              +   0.486495*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num17(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.910180  ModeIncrem 1 */
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
                              +   0.000060*(In[-4]+In[4])
                              +   0.002947*(In[-3]+In[3])
                              +   0.046647*(In[-2]+In[2])
                              +   0.241710*(In[-1]+In[1])
                              +   0.417273*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num18(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.082392  ModeIncrem 1 */
class cConvolSpec_REAL4_Num19 : public cConvolSpec<REAL4>
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
                              +   0.000016*(In[-5]+In[5])
                              +   0.000595*(In[-4]+In[4])
                              +   0.009837*(In[-3]+In[3])
                              +   0.072437*(In[-2]+In[2])
                              +   0.239165*(In[-1]+In[1])
                              +   0.355902*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num19(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.287189  ModeIncrem 1 */
class cConvolSpec_REAL4_Num20 : public cConvolSpec<REAL4>
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
                              +   0.000226*(In[-5]+In[5])
                              +   0.003035*(In[-4]+In[4])
                              +   0.022778*(In[-3]+In[3])
                              +   0.095882*(In[-2]+In[2])
                              +   0.226911*(In[-1]+In[1])
                              +   0.302334*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num20(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.530734  ModeIncrem 1 */
class cConvolSpec_REAL4_Num21 : public cConvolSpec<REAL4>
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
                              +   0.000152*(In[-6]+In[6])
                              +   0.001478*(In[-5]+In[5])
                              +   0.009469*(In[-4]+In[4])
                              +   0.040097*(In[-3]+In[3])
                              +   0.112349*(In[-2]+In[2])
                              +   0.208417*(In[-1]+In[1])
                              +   0.256076*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num21(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 0.904400  ModeIncrem 1 */
class cConvolSpec_REAL4_Num22 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num22(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.038883  ModeIncrem 1 */
class cConvolSpec_REAL4_Num23 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num23(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.193364  ModeIncrem 1 */
class cConvolSpec_REAL4_Num24 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num24(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.370815  ModeIncrem 1 */
class cConvolSpec_REAL4_Num25 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num25(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.574653  ModeIncrem 1 */
class cConvolSpec_REAL4_Num26 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num26(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.808801  ModeIncrem 1 */
class cConvolSpec_REAL4_Num27 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num27(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.077767  ModeIncrem 1 */
class cConvolSpec_REAL4_Num28 : public cConvolSpec<REAL4>
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

      cConvolSpec_REAL4_Num28(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 0.565250  ModeIncrem 1 */
class cConvolSpec_REAL4_Num29 : public cConvolSpec<REAL4>
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
                              +   0.000005*(In[-3]+In[3])
                              +   0.003967*(In[-2]+In[2])
                              +   0.184154*(In[-1]+In[1])
                              +   0.623749*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num29(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.649302  ModeIncrem 1 */
class cConvolSpec_REAL4_Num30 : public cConvolSpec<REAL4>
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
                              +   0.000059*(In[-3]+In[3])
                              +   0.010366*(In[-2]+In[2])
                              +   0.210157*(In[-1]+In[1])
                              +   0.558837*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num30(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.745852  ModeIncrem 1 */
class cConvolSpec_REAL4_Num31 : public cConvolSpec<REAL4>
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
                              +   0.000399*(In[-3]+In[3])
                              +   0.021738*(In[-2]+In[2])
                              +   0.229136*(In[-1]+In[1])
                              +   0.497455*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num31(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-3),-3,3,15,false)       {
      }
};

/* Sigma 0.856759  ModeIncrem 1 */
class cConvolSpec_REAL4_Num32 : public cConvolSpec<REAL4>
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
                              +   0.000022*(In[-4]+In[4])
                              +   0.001737*(In[-3]+In[3])
                              +   0.038213*(In[-2]+In[2])
                              +   0.239749*(In[-1]+In[1])
                              +   0.440557*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num32(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 0.984158  ModeIncrem 1 */
class cConvolSpec_REAL4_Num33 : public cConvolSpec<REAL4>
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
                              +   0.000185*(In[-4]+In[4])
                              +   0.005347*(In[-3]+In[3])
                              +   0.058183*(In[-2]+In[2])
                              +   0.241976*(In[-1]+In[1])
                              +   0.388617*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num33(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

/* Sigma 1.130501  ModeIncrem 1 */
class cConvolSpec_REAL4_Num34 : public cConvolSpec<REAL4>
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
                              +   0.000034*(In[-5]+In[5])
                              +   0.000946*(In[-4]+In[4])
                              +   0.012518*(In[-3]+In[3])
                              +   0.078766*(In[-2]+In[2])
                              +   0.236867*(In[-1]+In[1])
                              +   0.341739*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num34(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.298604  ModeIncrem 1 */
class cConvolSpec_REAL4_Num35 : public cConvolSpec<REAL4>
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
                              +   0.000253*(In[-5]+In[5])
                              +   0.003251*(In[-4]+In[4])
                              +   0.023584*(In[-3]+In[3])
                              +   0.096918*(In[-2]+In[2])
                              +   0.226091*(In[-1]+In[1])
                              +   0.299807*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num35(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 2.015874  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num36 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-8]+In[8])
                              +   17*(In[-7]+In[7])
                              +   84*(In[-6]+In[6])
                              +   315*(In[-5]+In[5])
                              +   933*(In[-4]+In[4])
                              +   2169*(In[-3]+In[3])
                              +   3963*(In[-2]+In[2])
                              +   5690*(In[-1]+In[1])
                              +   6420*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num36(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.539842  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num37 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-10]+In[10])
                              +   10*(In[-9]+In[9])
                              +   38*(In[-8]+In[8])
                              +   120*(In[-7]+In[7])
                              +   325*(In[-6]+In[6])
                              +   755*(In[-5]+In[5])
                              +   1503*(In[-4]+In[4])
                              +   2569*(In[-3]+In[3])
                              +   3766*(In[-2]+In[2])
                              +   4738*(In[-1]+In[1])
                              +   5114*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num37(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.200000  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num38 : public cConvolSpec<U_INT2>
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
                              +   4*(In[-12]+In[12])
                              +   12*(In[-11]+In[11])
                              +   32*(In[-10]+In[10])
                              +   80*(In[-9]+In[9])
                              +   183*(In[-8]+In[8])
                              +   379*(In[-7]+In[7])
                              +   712*(In[-6]+In[6])
                              +   1212*(In[-5]+In[5])
                              +   1875*(In[-4]+In[4])
                              +   2631*(In[-3]+In[3])
                              +   3352*(In[-2]+In[2])
                              +   3877*(In[-1]+In[1])
                              +   4070*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num38(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 4.031747  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num39 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-15]+In[15])
                              +   8*(In[-14]+In[14])
                              +   18*(In[-13]+In[13])
                              +   39*(In[-12]+In[12])
                              +   80*(In[-11]+In[11])
                              +   152*(In[-10]+In[10])
                              +   271*(In[-9]+In[9])
                              +   456*(In[-8]+In[8])
                              +   722*(In[-7]+In[7])
                              +   1075*(In[-6]+In[6])
                              +   1505*(In[-5]+In[5])
                              +   1982*(In[-4]+In[4])
                              +   2456*(In[-3]+In[3])
                              +   2862*(In[-2]+In[2])
                              +   3137*(In[-1]+In[1])
                              +   3236*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num39(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-15),-15,15,15,false)       {
      }
};

/* Sigma 5.079683  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num40 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-19]+In[19])
                              +   5*(In[-18]+In[18])
                              +   10*(In[-17]+In[17])
                              +   18*(In[-16]+In[16])
                              +   33*(In[-15]+In[15])
                              +   58*(In[-14]+In[14])
                              +   98*(In[-13]+In[13])
                              +   159*(In[-12]+In[12])
                              +   248*(In[-11]+In[11])
                              +   372*(In[-10]+In[10])
                              +   538*(In[-9]+In[9])
                              +   746*(In[-8]+In[8])
                              +   997*(In[-7]+In[7])
                              +   1282*(In[-6]+In[6])
                              +   1586*(In[-5]+In[5])
                              +   1887*(In[-4]+In[4])
                              +   2160*(In[-3]+In[3])
                              +   2379*(In[-2]+In[2])
                              +   2521*(In[-1]+In[1])
                              +   2570*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num40(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-19),-19,19,15,false)       {
      }
};

/* Sigma 1.259921  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num41 : public cConvolSpec<U_INT2>
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
                              +   6*(In[-5]+In[5])
                              +   84*(In[-4]+In[4])
                              +   684*(In[-3]+In[3])
                              +   3057*(In[-2]+In[2])
                              +   7498*(In[-1]+In[1])
                              +   10110*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num41(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.587401  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num42 : public cConvolSpec<U_INT2>
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
                              +   8*(In[-6]+In[6])
                              +   66*(In[-5]+In[5])
                              +   375*(In[-4]+In[4])
                              +   1439*(In[-3]+In[3])
                              +   3759*(In[-2]+In[2])
                              +   6686*(In[-1]+In[1])
                              +   8102*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num42(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 2.000000  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num43 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-8]+In[8])
                              +   16*(In[-7]+In[7])
                              +   79*(In[-6]+In[6])
                              +   303*(In[-5]+In[5])
                              +   912*(In[-4]+In[4])
                              +   2149*(In[-3]+In[3])
                              +   3964*(In[-2]+In[2])
                              +   5724*(In[-1]+In[1])
                              +   6468*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num43(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.519842  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num44 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-10]+In[10])
                              +   10*(In[-9]+In[9])
                              +   36*(In[-8]+In[8])
                              +   114*(In[-7]+In[7])
                              +   314*(In[-6]+In[6])
                              +   738*(In[-5]+In[5])
                              +   1486*(In[-4]+In[4])
                              +   2561*(In[-3]+In[3])
                              +   3777*(In[-2]+In[2])
                              +   4769*(In[-1]+In[1])
                              +   5154*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num44(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.174802  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num45 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-12]+In[12])
                              +   11*(In[-11]+In[11])
                              +   30*(In[-10]+In[10])
                              +   76*(In[-9]+In[9])
                              +   176*(In[-8]+In[8])
                              +   368*(In[-7]+In[7])
                              +   698*(In[-6]+In[6])
                              +   1199*(In[-5]+In[5])
                              +   1866*(In[-4]+In[4])
                              +   2634*(In[-3]+In[3])
                              +   3368*(In[-2]+In[2])
                              +   3904*(In[-1]+In[1])
                              +   4102*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num45(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 1.902731  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num46 : public cConvolSpec<U_INT2>
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
                              +   1*(In[-8]+In[8])
                              +   9*(In[-7]+In[7])
                              +   53*(In[-6]+In[6])
                              +   232*(In[-5]+In[5])
                              +   783*(In[-4]+In[4])
                              +   2016*(In[-3]+In[3])
                              +   3959*(In[-2]+In[2])
                              +   5935*(In[-1]+In[1])
                              +   6792*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num46(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.262742  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num47 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-9]+In[9])
                              +   12*(In[-8]+In[8])
                              +   52*(In[-7]+In[7])
                              +   180*(In[-6]+In[6])
                              +   519*(In[-5]+In[5])
                              +   1232*(In[-4]+In[4])
                              +   2414*(In[-3]+In[3])
                              +   3902*(In[-2]+In[2])
                              +   5206*(In[-1]+In[1])
                              +   5730*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num47(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.690869  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num48 : public cConvolSpec<U_INT2>
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
                              +   1*(In[-11]+In[11])
                              +   5*(In[-10]+In[10])
                              +   19*(In[-9]+In[9])
                              +   61*(In[-8]+In[8])
                              +   170*(In[-7]+In[7])
                              +   414*(In[-6]+In[6])
                              +   877*(In[-5]+In[5])
                              +   1620*(In[-4]+In[4])
                              +   2613*(In[-3]+In[3])
                              +   3676*(In[-2]+In[2])
                              +   4512*(In[-1]+In[1])
                              +   4832*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num48(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 3.805463  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num49 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-15]+In[15])
                              +   4*(In[-14]+In[14])
                              +   10*(In[-13]+In[13])
                              +   24*(In[-12]+In[12])
                              +   54*(In[-11]+In[11])
                              +   111*(In[-10]+In[10])
                              +   212*(In[-9]+In[9])
                              +   381*(In[-8]+In[8])
                              +   637*(In[-7]+In[7])
                              +   995*(In[-6]+In[6])
                              +   1452*(In[-5]+In[5])
                              +   1978*(In[-4]+In[4])
                              +   2515*(In[-3]+In[3])
                              +   2986*(In[-2]+In[2])
                              +   3310*(In[-1]+In[1])
                              +   3426*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num49(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-15),-15,15,15,false)       {
      }
};

/* Sigma 4.525483  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num50 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-17]+In[17])
                              +   6*(In[-16]+In[16])
                              +   12*(In[-15]+In[15])
                              +   25*(In[-14]+In[14])
                              +   47*(In[-13]+In[13])
                              +   87*(In[-12]+In[12])
                              +   152*(In[-11]+In[11])
                              +   253*(In[-10]+In[10])
                              +   402*(In[-9]+In[9])
                              +   608*(In[-8]+In[8])
                              +   876*(In[-7]+In[7])
                              +   1201*(In[-6]+In[6])
                              +   1570*(In[-5]+In[5])
                              +   1954*(In[-4]+In[4])
                              +   2316*(In[-3]+In[3])
                              +   2616*(In[-2]+In[2])
                              +   2814*(In[-1]+In[1])
                              +   2884*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num50(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-17),-17,17,15,false)       {
      }
};

/* Sigma 1.189207  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num51 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-5]+In[5])
                              +   51*(In[-4]+In[4])
                              +   529*(In[-3]+In[3])
                              +   2812*(In[-2]+In[2])
                              +   7651*(In[-1]+In[1])
                              +   10678*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num51(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.414214  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num52 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-6]+In[6])
                              +   22*(In[-5]+In[5])
                              +   194*(In[-4]+In[4])
                              +   1045*(In[-3]+In[3])
                              +   3469*(In[-2]+In[2])
                              +   7124*(In[-1]+In[1])
                              +   9056*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num52(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.681793  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num53 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-7]+In[7])
                              +   16*(In[-6]+In[6])
                              +   104*(In[-5]+In[5])
                              +   491*(In[-4]+In[4])
                              +   1634*(In[-3]+In[3])
                              +   3855*(In[-2]+In[2])
                              +   6452*(In[-1]+In[1])
                              +   7660*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num53(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.378414  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num54 : public cConvolSpec<U_INT2>
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
                              +   5*(In[-9]+In[9])
                              +   21*(In[-8]+In[8])
                              +   76*(In[-7]+In[7])
                              +   237*(In[-6]+In[6])
                              +   618*(In[-5]+In[5])
                              +   1354*(In[-4]+In[4])
                              +   2492*(In[-3]+In[3])
                              +   3851*(In[-2]+In[2])
                              +   5001*(In[-1]+In[1])
                              +   5458*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num54(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.828427  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num55 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-11]+In[11])
                              +   9*(In[-10]+In[10])
                              +   31*(In[-9]+In[9])
                              +   88*(In[-8]+In[8])
                              +   222*(In[-7]+In[7])
                              +   496*(In[-6]+In[6])
                              +   980*(In[-5]+In[5])
                              +   1709*(In[-4]+In[4])
                              +   2635*(In[-3]+In[3])
                              +   3590*(In[-2]+In[2])
                              +   4322*(In[-1]+In[1])
                              +   4598*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num55(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 1.837917  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num56 : public cConvolSpec<U_INT2>
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
                              +   6*(In[-7]+In[7])
                              +   39*(In[-6]+In[6])
                              +   190*(In[-5]+In[5])
                              +   697*(In[-4]+In[4])
                              +   1915*(In[-3]+In[3])
                              +   3943*(In[-2]+In[2])
                              +   6081*(In[-1]+In[1])
                              +   7026*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num56(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.111213  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num57 : public cConvolSpec<U_INT2>
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
                              +   5*(In[-8]+In[8])
                              +   28*(In[-7]+In[7])
                              +   116*(In[-6]+In[6])
                              +   391*(In[-5]+In[5])
                              +   1054*(In[-4]+In[4])
                              +   2277*(In[-3]+In[3])
                              +   3950*(In[-2]+In[2])
                              +   5495*(In[-1]+In[1])
                              +   6136*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num57(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.425147  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num58 : public cConvolSpec<U_INT2>
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
                              +   6*(In[-9]+In[9])
                              +   25*(In[-8]+In[8])
                              +   88*(In[-7]+In[7])
                              +   262*(In[-6]+In[6])
                              +   658*(In[-5]+In[5])
                              +   1400*(In[-4]+In[4])
                              +   2517*(In[-3]+In[3])
                              +   3828*(In[-2]+In[2])
                              +   4922*(In[-1]+In[1])
                              +   5354*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num58(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 2.785762  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num59 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-11]+In[11])
                              +   8*(In[-10]+In[10])
                              +   27*(In[-9]+In[9])
                              +   79*(In[-8]+In[8])
                              +   205*(In[-7]+In[7])
                              +   470*(In[-6]+In[6])
                              +   949*(In[-5]+In[5])
                              +   1683*(In[-4]+In[4])
                              +   2630*(In[-3]+In[3])
                              +   3617*(In[-2]+In[2])
                              +   4380*(In[-1]+In[1])
                              +   4668*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num59(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 3.675835  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num60 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-14]+In[14])
                              +   7*(In[-13]+In[13])
                              +   18*(In[-12]+In[12])
                              +   41*(In[-11]+In[11])
                              +   90*(In[-10]+In[10])
                              +   180*(In[-9]+In[9])
                              +   337*(In[-8]+In[8])
                              +   585*(In[-7]+In[7])
                              +   943*(In[-6]+In[6])
                              +   1414*(In[-5]+In[5])
                              +   1968*(In[-4]+In[4])
                              +   2546*(In[-3]+In[3])
                              +   3061*(In[-2]+In[2])
                              +   3418*(In[-1]+In[1])
                              +   3546*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num60(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-14),-14,14,15,false)       {
      }
};

/* Sigma 4.222425  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num61 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-16]+In[16])
                              +   6*(In[-15]+In[15])
                              +   13*(In[-14]+In[14])
                              +   28*(In[-13]+In[13])
                              +   55*(In[-12]+In[12])
                              +   105*(In[-11]+In[11])
                              +   189*(In[-10]+In[10])
                              +   322*(In[-9]+In[9])
                              +   518*(In[-8]+In[8])
                              +   787*(In[-7]+In[7])
                              +   1131*(In[-6]+In[6])
                              +   1537*(In[-5]+In[5])
                              +   1976*(In[-4]+In[4])
                              +   2403*(In[-3]+In[3])
                              +   2763*(In[-2]+In[2])
                              +   3004*(In[-1]+In[1])
                              +   3090*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num61(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-16),-16,16,15,false)       {
      }
};

/* Sigma 1.148698  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num62 : public cConvolSpec<U_INT2>
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
                              +   2*(In[-5]+In[5])
                              +   36*(In[-4]+In[4])
                              +   446*(In[-3]+In[3])
                              +   2655*(In[-2]+In[2])
                              +   7729*(In[-1]+In[1])
                              +   11032*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num62(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.319508  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num63 : public cConvolSpec<U_INT2>
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
                              +   10*(In[-5]+In[5])
                              +   120*(In[-4]+In[4])
                              +   821*(In[-3]+In[3])
                              +   3236*(In[-2]+In[2])
                              +   7359*(In[-1]+In[1])
                              +   9676*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num63(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.515717  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num64 : public cConvolSpec<U_INT2>
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
                              +   4*(In[-6]+In[6])
                              +   44*(In[-5]+In[5])
                              +   294*(In[-4]+In[4])
                              +   1280*(In[-3]+In[3])
                              +   3658*(In[-2]+In[2])
                              +   6868*(In[-1]+In[1])
                              +   8472*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num64(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.741101  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num65 : public cConvolSpec<U_INT2>
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
                              +   23*(In[-6]+In[6])
                              +   134*(In[-5]+In[5])
                              +   568*(In[-4]+In[4])
                              +   1747*(In[-3]+In[3])
                              +   3898*(In[-2]+In[2])
                              +   6308*(In[-1]+In[1])
                              +   7406*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num65(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.297397  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num66 : public cConvolSpec<U_INT2>
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
                              +   3*(In[-9]+In[9])
                              +   14*(In[-8]+In[8])
                              +   58*(In[-7]+In[7])
                              +   197*(In[-6]+In[6])
                              +   549*(In[-5]+In[5])
                              +   1270*(In[-4]+In[4])
                              +   2439*(In[-3]+In[3])
                              +   3888*(In[-2]+In[2])
                              +   5143*(In[-1]+In[1])
                              +   5646*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num66(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.639016  ModeIncrem 0 */
class cConvolSpec_U_INT2_Num67 : public cConvolSpec<U_INT2>
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
                              +   4*(In[-10]+In[10])
                              +   16*(In[-9]+In[9])
                              +   52*(In[-8]+In[8])
                              +   152*(In[-7]+In[7])
                              +   383*(In[-6]+In[6])
                              +   836*(In[-5]+In[5])
                              +   1583*(In[-4]+In[4])
                              +   2601*(In[-3]+In[3])
                              +   3708*(In[-2]+In[2])
                              +   4587*(In[-1]+In[1])
                              +   4924*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num67(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 2.015874  ModeIncrem 0 */
class cConvolSpec_REAL4_Num36 : public cConvolSpec<REAL4>
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
                              +   0.000087*(In[-8]+In[8])
                              +   0.000532*(In[-7]+In[7])
                              +   0.002551*(In[-6]+In[6])
                              +   0.009615*(In[-5]+In[5])
                              +   0.028463*(In[-4]+In[4])
                              +   0.066196*(In[-3]+In[3])
                              +   0.120954*(In[-2]+In[2])
                              +   0.173653*(In[-1]+In[1])
                              +   0.195899*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num36(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.539842  ModeIncrem 0 */
class cConvolSpec_REAL4_Num37 : public cConvolSpec<REAL4>
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
                              +   0.000074*(In[-10]+In[10])
                              +   0.000317*(In[-9]+In[9])
                              +   0.001165*(In[-8]+In[8])
                              +   0.003671*(In[-7]+In[7])
                              +   0.009929*(In[-6]+In[6])
                              +   0.023042*(In[-5]+In[5])
                              +   0.045879*(In[-4]+In[4])
                              +   0.078387*(In[-3]+In[3])
                              +   0.114922*(In[-2]+In[2])
                              +   0.144576*(In[-1]+In[1])
                              +   0.156073*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num37(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.200000  ModeIncrem 0 */
class cConvolSpec_REAL4_Num38 : public cConvolSpec<REAL4>
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
                              +   0.000116*(In[-12]+In[12])
                              +   0.000354*(In[-11]+In[11])
                              +   0.000978*(In[-10]+In[10])
                              +   0.002456*(In[-9]+In[9])
                              +   0.005595*(In[-8]+In[8])
                              +   0.011570*(In[-7]+In[7])
                              +   0.021717*(In[-6]+In[6])
                              +   0.036998*(In[-5]+In[5])
                              +   0.057212*(In[-4]+In[4])
                              +   0.080303*(In[-3]+In[3])
                              +   0.102307*(In[-2]+In[2])
                              +   0.118306*(In[-1]+In[1])
                              +   0.124177*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num38(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 4.031747  ModeIncrem 0 */
class cConvolSpec_REAL4_Num39 : public cConvolSpec<REAL4>
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
                              +   0.000101*(In[-15]+In[15])
                              +   0.000245*(In[-14]+In[14])
                              +   0.000560*(In[-13]+In[13])
                              +   0.001204*(In[-12]+In[12])
                              +   0.002433*(In[-11]+In[11])
                              +   0.004627*(In[-10]+In[10])
                              +   0.008276*(In[-9]+In[9])
                              +   0.013924*(In[-8]+In[8])
                              +   0.022035*(In[-7]+In[7])
                              +   0.032802*(In[-6]+In[6])
                              +   0.045930*(In[-5]+In[5])
                              +   0.060494*(In[-4]+In[4])
                              +   0.074945*(In[-3]+In[3])
                              +   0.087337*(In[-2]+In[2])
                              +   0.095735*(In[-1]+In[1])
                              +   0.098710*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num39(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-15),-15,15,15,false)       {
      }
};

/* Sigma 5.079683  ModeIncrem 0 */
class cConvolSpec_REAL4_Num40 : public cConvolSpec<REAL4>
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
                              +   0.000073*(In[-19]+In[19])
                              +   0.000150*(In[-18]+In[18])
                              +   0.000295*(In[-17]+In[17])
                              +   0.000558*(In[-16]+In[16])
                              +   0.001016*(In[-15]+In[15])
                              +   0.001779*(In[-14]+In[14])
                              +   0.002998*(In[-13]+In[13])
                              +   0.004858*(In[-12]+In[12])
                              +   0.007576*(In[-11]+In[11])
                              +   0.011365*(In[-10]+In[10])
                              +   0.016404*(In[-9]+In[9])
                              +   0.022781*(In[-8]+In[8])
                              +   0.030436*(In[-7]+In[7])
                              +   0.039124*(In[-6]+In[6])
                              +   0.048386*(In[-5]+In[5])
                              +   0.057572*(In[-4]+In[4])
                              +   0.065907*(In[-3]+In[3])
                              +   0.072590*(In[-2]+In[2])
                              +   0.076920*(In[-1]+In[1])
                              +   0.078420*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num40(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-19),-19,19,15,false)       {
      }
};

/* Sigma 1.259921  ModeIncrem 0 */
class cConvolSpec_REAL4_Num41 : public cConvolSpec<REAL4>
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
                              +   0.000171*(In[-5]+In[5])
                              +   0.002557*(In[-4]+In[4])
                              +   0.020874*(In[-3]+In[3])
                              +   0.093295*(In[-2]+In[2])
                              +   0.228832*(In[-1]+In[1])
                              +   0.308543*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num41(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.587401  ModeIncrem 0 */
class cConvolSpec_REAL4_Num42 : public cConvolSpec<REAL4>
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
                              +   0.000244*(In[-6]+In[6])
                              +   0.002027*(In[-5]+In[5])
                              +   0.011438*(In[-4]+In[4])
                              +   0.043906*(In[-3]+In[3])
                              +   0.114708*(In[-2]+In[2])
                              +   0.204057*(In[-1]+In[1])
                              +   0.247243*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num42(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 2.000000  ModeIncrem 0 */
class cConvolSpec_REAL4_Num43 : public cConvolSpec<REAL4>
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
                              +   0.000078*(In[-8]+In[8])
                              +   0.000488*(In[-7]+In[7])
                              +   0.002402*(In[-6]+In[6])
                              +   0.009244*(In[-5]+In[5])
                              +   0.027833*(In[-4]+In[4])
                              +   0.065590*(In[-3]+In[3])
                              +   0.120980*(In[-2]+In[2])
                              +   0.174673*(In[-1]+In[1])
                              +   0.197421*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num43(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.519842  ModeIncrem 0 */
class cConvolSpec_REAL4_Num44 : public cConvolSpec<REAL4>
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
                              +   0.000066*(In[-10]+In[10])
                              +   0.000290*(In[-9]+In[9])
                              +   0.001087*(In[-8]+In[8])
                              +   0.003488*(In[-7]+In[7])
                              +   0.009583*(In[-6]+In[6])
                              +   0.022534*(In[-5]+In[5])
                              +   0.045357*(In[-4]+In[4])
                              +   0.078150*(In[-3]+In[3])
                              +   0.115266*(In[-2]+In[2])
                              +   0.145533*(In[-1]+In[1])
                              +   0.157295*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num44(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 3.174802  ModeIncrem 0 */
class cConvolSpec_REAL4_Num45 : public cConvolSpec<REAL4>
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
                              +   0.000105*(In[-12]+In[12])
                              +   0.000325*(In[-11]+In[11])
                              +   0.000913*(In[-10]+In[10])
                              +   0.002326*(In[-9]+In[9])
                              +   0.005369*(In[-8]+In[8])
                              +   0.011232*(In[-7]+In[7])
                              +   0.021292*(In[-6]+In[6])
                              +   0.036582*(In[-5]+In[5])
                              +   0.056960*(In[-4]+In[4])
                              +   0.080379*(In[-3]+In[3])
                              +   0.102796*(In[-2]+In[2])
                              +   0.119145*(In[-1]+In[1])
                              +   0.125153*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num45(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

/* Sigma 1.902731  ModeIncrem 0 */
class cConvolSpec_REAL4_Num46 : public cConvolSpec<REAL4>
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
                              +   0.000036*(In[-8]+In[8])
                              +   0.000277*(In[-7]+In[7])
                              +   0.001605*(In[-6]+In[6])
                              +   0.007091*(In[-5]+In[5])
                              +   0.023907*(In[-4]+In[4])
                              +   0.061514*(In[-3]+In[3])
                              +   0.120810*(In[-2]+In[2])
                              +   0.181116*(In[-1]+In[1])
                              +   0.207287*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num46(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.262742  ModeIncrem 0 */
class cConvolSpec_REAL4_Num47 : public cConvolSpec<REAL4>
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
                              +   0.000073*(In[-9]+In[9])
                              +   0.000373*(In[-8]+In[8])
                              +   0.001576*(In[-7]+In[7])
                              +   0.005499*(In[-6]+In[6])
                              +   0.015829*(In[-5]+In[5])
                              +   0.037591*(In[-4]+In[4])
                              +   0.073656*(In[-3]+In[3])
                              +   0.119086*(In[-2]+In[2])
                              +   0.158871*(In[-1]+In[1])
                              +   0.174893*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num47(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.690869  ModeIncrem 0 */
class cConvolSpec_REAL4_Num48 : public cConvolSpec<REAL4>
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
                              +   0.000038*(In[-11]+In[11])
                              +   0.000160*(In[-10]+In[10])
                              +   0.000585*(In[-9]+In[9])
                              +   0.001866*(In[-8]+In[8])
                              +   0.005197*(In[-7]+In[7])
                              +   0.012624*(In[-6]+In[6])
                              +   0.026751*(In[-5]+In[5])
                              +   0.049451*(In[-4]+In[4])
                              +   0.079747*(In[-3]+In[3])
                              +   0.112189*(In[-2]+In[2])
                              +   0.137687*(In[-1]+In[1])
                              +   0.147414*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num48(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 3.805463  ModeIncrem 0 */
class cConvolSpec_REAL4_Num49 : public cConvolSpec<REAL4>
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
                              +   0.000046*(In[-15]+In[15])
                              +   0.000125*(In[-14]+In[14])
                              +   0.000316*(In[-13]+In[13])
                              +   0.000745*(In[-12]+In[12])
                              +   0.001642*(In[-11]+In[11])
                              +   0.003376*(In[-10]+In[10])
                              +   0.006481*(In[-9]+In[9])
                              +   0.011617*(In[-8]+In[8])
                              +   0.019442*(In[-7]+In[7])
                              +   0.030378*(In[-6]+In[6])
                              +   0.044315*(In[-5]+In[5])
                              +   0.060358*(In[-4]+In[4])
                              +   0.076753*(In[-3]+In[3])
                              +   0.091126*(In[-2]+In[2])
                              +   0.101011*(In[-1]+In[1])
                              +   0.104539*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num49(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-15),-15,15,15,false)       {
      }
};

/* Sigma 4.525483  ModeIncrem 0 */
class cConvolSpec_REAL4_Num50 : public cConvolSpec<REAL4>
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
                              +   0.000078*(In[-17]+In[17])
                              +   0.000174*(In[-16]+In[16])
                              +   0.000370*(In[-15]+In[15])
                              +   0.000749*(In[-14]+In[14])
                              +   0.001445*(In[-13]+In[13])
                              +   0.002653*(In[-12]+In[12])
                              +   0.004641*(In[-11]+In[11])
                              +   0.007734*(In[-10]+In[10])
                              +   0.012276*(In[-9]+In[9])
                              +   0.018560*(In[-8]+In[8])
                              +   0.026728*(In[-7]+In[7])
                              +   0.036665*(In[-6]+In[6])
                              +   0.047909*(In[-5]+In[5])
                              +   0.059628*(In[-4]+In[4])
                              +   0.070692*(In[-3]+In[3])
                              +   0.079831*(In[-2]+In[2])
                              +   0.085872*(In[-1]+In[1])
                              +   0.087986*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num50(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-17),-17,17,15,false)       {
      }
};

/* Sigma 1.189207  ModeIncrem 0 */
class cConvolSpec_REAL4_Num51 : public cConvolSpec<REAL4>
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
                              +   0.000075*(In[-5]+In[5])
                              +   0.001546*(In[-4]+In[4])
                              +   0.016137*(In[-3]+In[3])
                              +   0.085818*(In[-2]+In[2])
                              +   0.233492*(In[-1]+In[1])
                              +   0.325864*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num51(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.414214  ModeIncrem 0 */
class cConvolSpec_REAL4_Num52 : public cConvolSpec<REAL4>
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
                              +   0.000048*(In[-6]+In[6])
                              +   0.000681*(In[-5]+In[5])
                              +   0.005931*(In[-4]+In[4])
                              +   0.031881*(In[-3]+In[3])
                              +   0.105868*(In[-2]+In[2])
                              +   0.217421*(In[-1]+In[1])
                              +   0.276340*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num52(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.681793  ModeIncrem 0 */
class cConvolSpec_REAL4_Num53 : public cConvolSpec<REAL4>
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
                              +   0.000051*(In[-7]+In[7])
                              +   0.000481*(In[-6]+In[6])
                              +   0.003191*(In[-5]+In[5])
                              +   0.014981*(In[-4]+In[4])
                              +   0.049858*(In[-3]+In[3])
                              +   0.117649*(In[-2]+In[2])
                              +   0.196902*(In[-1]+In[1])
                              +   0.233773*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num53(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.378414  ModeIncrem 0 */
class cConvolSpec_REAL4_Num54 : public cConvolSpec<REAL4>
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
                              +   0.000143*(In[-9]+In[9])
                              +   0.000631*(In[-8]+In[8])
                              +   0.002332*(In[-7]+In[7])
                              +   0.007237*(In[-6]+In[6])
                              +   0.018869*(In[-5]+In[5])
                              +   0.041326*(In[-4]+In[4])
                              +   0.076037*(In[-3]+In[3])
                              +   0.117534*(In[-2]+In[2])
                              +   0.152631*(In[-1]+In[1])
                              +   0.166521*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num54(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.828427  ModeIncrem 0 */
class cConvolSpec_REAL4_Num55 : public cConvolSpec<REAL4>
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
                              +   0.000079*(In[-11]+In[11])
                              +   0.000289*(In[-10]+In[10])
                              +   0.000935*(In[-9]+In[9])
                              +   0.002678*(In[-8]+In[8])
                              +   0.006773*(In[-7]+In[7])
                              +   0.015137*(In[-6]+In[6])
                              +   0.029892*(In[-5]+In[5])
                              +   0.052158*(In[-4]+In[4])
                              +   0.080421*(In[-3]+In[3])
                              +   0.109568*(In[-2]+In[2])
                              +   0.131908*(In[-1]+In[1])
                              +   0.140325*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num55(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 1.837917  ModeIncrem 0 */
class cConvolSpec_REAL4_Num56 : public cConvolSpec<REAL4>
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
                              +   0.000180*(In[-7]+In[7])
                              +   0.001181*(In[-6]+In[6])
                              +   0.005790*(In[-5]+In[5])
                              +   0.021258*(In[-4]+In[4])
                              +   0.058444*(In[-3]+In[3])
                              +   0.120337*(In[-2]+In[2])
                              +   0.185595*(In[-1]+In[1])
                              +   0.214430*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num56(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.111213  ModeIncrem 0 */
class cConvolSpec_REAL4_Num57 : public cConvolSpec<REAL4>
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
                              +   0.000162*(In[-8]+In[8])
                              +   0.000848*(In[-7]+In[7])
                              +   0.003552*(In[-6]+In[6])
                              +   0.011933*(In[-5]+In[5])
                              +   0.032153*(In[-4]+In[4])
                              +   0.069501*(In[-3]+In[3])
                              +   0.120530*(In[-2]+In[2])
                              +   0.167707*(In[-1]+In[1])
                              +   0.187226*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num57(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

/* Sigma 2.425147  ModeIncrem 0 */
class cConvolSpec_REAL4_Num58 : public cConvolSpec<REAL4>
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
                              +   0.000037*(In[-10]+In[10])
                              +   0.000184*(In[-9]+In[9])
                              +   0.000764*(In[-8]+In[8])
                              +   0.002686*(In[-7]+In[7])
                              +   0.007989*(In[-6]+In[6])
                              +   0.020090*(In[-5]+In[5])
                              +   0.042722*(In[-4]+In[4])
                              +   0.076822*(In[-3]+In[3])
                              +   0.116817*(In[-2]+In[2])
                              +   0.150216*(In[-1]+In[1])
                              +   0.163349*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num58(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

/* Sigma 2.785762  ModeIncrem 0 */
class cConvolSpec_REAL4_Num59 : public cConvolSpec<REAL4>
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
                              +   0.000064*(In[-11]+In[11])
                              +   0.000243*(In[-10]+In[10])
                              +   0.000815*(In[-9]+In[9])
                              +   0.002409*(In[-8]+In[8])
                              +   0.006268*(In[-7]+In[7])
                              +   0.014356*(In[-6]+In[6])
                              +   0.028945*(In[-5]+In[5])
                              +   0.051372*(In[-4]+In[4])
                              +   0.080263*(In[-3]+In[3])
                              +   0.110390*(In[-2]+In[2])
                              +   0.133653*(In[-1]+In[1])
                              +   0.142449*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num59(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-11),-11,11,15,false)       {
      }
};

/* Sigma 3.675835  ModeIncrem 0 */
class cConvolSpec_REAL4_Num60 : public cConvolSpec<REAL4>
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
                              +   0.000080*(In[-14]+In[14])
                              +   0.000216*(In[-13]+In[13])
                              +   0.000542*(In[-12]+In[12])
                              +   0.001263*(In[-11]+In[11])
                              +   0.002735*(In[-10]+In[10])
                              +   0.005501*(In[-9]+In[9])
                              +   0.010280*(In[-8]+In[8])
                              +   0.017848*(In[-7]+In[7])
                              +   0.028790*(In[-6]+In[6])
                              +   0.043146*(In[-5]+In[5])
                              +   0.060075*(In[-4]+In[4])
                              +   0.077715*(In[-3]+In[3])
                              +   0.093404*(In[-2]+In[2])
                              +   0.104299*(In[-1]+In[1])
                              +   0.108207*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num60(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-14),-14,14,15,false)       {
      }
};

/* Sigma 4.222425  ModeIncrem 0 */
class cConvolSpec_REAL4_Num61 : public cConvolSpec<REAL4>
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
                              +   0.000074*(In[-16]+In[16])
                              +   0.000176*(In[-15]+In[15])
                              +   0.000397*(In[-14]+In[14])
                              +   0.000843*(In[-13]+In[13])
                              +   0.001693*(In[-12]+In[12])
                              +   0.003217*(In[-11]+In[11])
                              +   0.005782*(In[-10]+In[10])
                              +   0.009827*(In[-9]+In[9])
                              +   0.015795*(In[-8]+In[8])
                              +   0.024008*(In[-7]+In[7])
                              +   0.034511*(In[-6]+In[6])
                              +   0.046914*(In[-5]+In[5])
                              +   0.060313*(In[-4]+In[4])
                              +   0.073328*(In[-3]+In[3])
                              +   0.084311*(In[-2]+In[2])
                              +   0.091676*(In[-1]+In[1])
                              +   0.094271*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num61(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-16),-16,16,15,false)       {
      }
};

/* Sigma 1.148698  ModeIncrem 0 */
class cConvolSpec_REAL4_Num62 : public cConvolSpec<REAL4>
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
                              +   0.000044*(In[-5]+In[5])
                              +   0.001110*(In[-4]+In[4])
                              +   0.013603*(In[-3]+In[3])
                              +   0.081032*(In[-2]+In[2])
                              +   0.235880*(In[-1]+In[1])
                              +   0.336661*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num62(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.319508  ModeIncrem 0 */
class cConvolSpec_REAL4_Num63 : public cConvolSpec<REAL4>
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
                              +   0.000309*(In[-5]+In[5])
                              +   0.003669*(In[-4]+In[4])
                              +   0.025070*(In[-3]+In[3])
                              +   0.098741*(In[-2]+In[2])
                              +   0.224569*(In[-1]+In[1])
                              +   0.295284*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num63(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

/* Sigma 1.515717  ModeIncrem 0 */
class cConvolSpec_REAL4_Num64 : public cConvolSpec<REAL4>
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
                              +   0.000133*(In[-6]+In[6])
                              +   0.001351*(In[-5]+In[5])
                              +   0.008972*(In[-4]+In[4])
                              +   0.039063*(In[-3]+In[3])
                              +   0.111642*(In[-2]+In[2])
                              +   0.209578*(In[-1]+In[1])
                              +   0.258522*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num64(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

/* Sigma 1.741101  ModeIncrem 0 */
class cConvolSpec_REAL4_Num65 : public cConvolSpec<REAL4>
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
                              +   0.000086*(In[-7]+In[7])
                              +   0.000697*(In[-6]+In[6])
                              +   0.004082*(In[-5]+In[5])
                              +   0.017327*(In[-4]+In[4])
                              +   0.053313*(In[-3]+In[3])
                              +   0.118957*(In[-2]+In[2])
                              +   0.192522*(In[-1]+In[1])
                              +   0.226032*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num65(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

/* Sigma 2.297397  ModeIncrem 0 */
class cConvolSpec_REAL4_Num66 : public cConvolSpec<REAL4>
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
                              +   0.000090*(In[-9]+In[9])
                              +   0.000440*(In[-8]+In[8])
                              +   0.001784*(In[-7]+In[7])
                              +   0.006000*(In[-6]+In[6])
                              +   0.016739*(In[-5]+In[5])
                              +   0.038750*(In[-4]+In[4])
                              +   0.074436*(In[-3]+In[3])
                              +   0.118655*(In[-2]+In[2])
                              +   0.156957*(In[-1]+In[1])
                              +   0.172298*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num66(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-9),-9,9,15,false)       {
      }
};

/* Sigma 2.639016  ModeIncrem 0 */
class cConvolSpec_REAL4_Num67 : public cConvolSpec<REAL4>
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
                              +   0.000125*(In[-10]+In[10])
                              +   0.000480*(In[-9]+In[9])
                              +   0.001603*(In[-8]+In[8])
                              +   0.004646*(In[-7]+In[7])
                              +   0.011687*(In[-6]+In[6])
                              +   0.025506*(In[-5]+In[5])
                              +   0.048300*(In[-4]+In[4])
                              +   0.079364*(In[-3]+In[3])
                              +   0.113155*(In[-2]+In[2])
                              +   0.139992*(In[-1]+In[1])
                              +   0.150284*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num67(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

