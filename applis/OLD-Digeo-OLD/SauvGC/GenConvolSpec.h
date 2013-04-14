#include "general/all.h"
#include "private/all.h"
#include "Digeo.h"

namespace NS_ParamDigeo {
class cConvolSpec_U_INT2_Num0_0_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num0_0_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num1_1_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num1_1_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num2_2_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num2_2_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-4),-4,4,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num3_3_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num3_3_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num4_4_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num4_4_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num5_5_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num5_5_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num6_6_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num6_6_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num7_7_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num7_7_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_REAL4_Num0_0_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num0_0_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_REAL4_Num1_1_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num1_1_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

class cConvolSpec_REAL4_Num2_2_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num2_2_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-4),-4,4,15,false)       {
      }
};

class cConvolSpec_REAL4_Num3_3_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num3_3_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

class cConvolSpec_REAL4_Num4_4_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num4_4_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_REAL4_Num5_5_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num5_5_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_REAL4_Num6_6_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num6_6_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-7),-7,7,15,false)       {
      }
};

class cConvolSpec_REAL4_Num7_7_5_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num7_7_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num8_0_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-7]+In[7])
                              +   9*(In[-6]+In[6])
                              +   71*(In[-5]+In[5])
                              +   390*(In[-4]+In[4])
                              +   1465*(In[-3]+In[3])
                              +   3774*(In[-2]+In[2])
                              +   6655*(In[-1]+In[1])
                              +   8038*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num8_0_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-7),-7,7,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num9_1_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-8]+In[8])
                              +   6*(In[-7]+In[7])
                              +   39*(In[-6]+In[6])
                              +   190*(In[-5]+In[5])
                              +   696*(In[-4]+In[4])
                              +   1915*(In[-3]+In[3])
                              +   3943*(In[-2]+In[2])
                              +   6081*(In[-1]+In[1])
                              +   7026*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num9_1_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num10_2_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-9]+In[9])
                              +   5*(In[-8]+In[8])
                              +   28*(In[-7]+In[7])
                              +   116*(In[-6]+In[6])
                              +   391*(In[-5]+In[5])
                              +   1054*(In[-4]+In[4])
                              +   2277*(In[-3]+In[3])
                              +   3949*(In[-2]+In[2])
                              +   5495*(In[-1]+In[1])
                              +   6136*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num10_2_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-9),-9,9,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num11_3_5_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num11_3_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num12_4_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-12]+In[12])
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
                              +   4379*(In[-1]+In[1])
                              +   4668*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num12_4_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num13_5_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-13]+In[13])
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
                              +   3876*(In[-1]+In[1])
                              +   4070*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num13_5_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-13),-13,13,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num14_6_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-15]+In[15])
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
                              +   3417*(In[-1]+In[1])
                              +   3546*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num14_6_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-15),-15,15,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num15_7_5_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-17]+In[17])
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
                              +   2762*(In[-2]+In[2])
                              +   3004*(In[-1]+In[1])
                              +   3090*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num15_7_5_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-17),-17,17,15,false)       {
      }
};

class cConvolSpec_REAL4_Num8_0_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000001*(In[-8]+In[8])
                              +   0.000023*(In[-7]+In[7])
                              +   0.000269*(In[-6]+In[6])
                              +   0.002164*(In[-5]+In[5])
                              +   0.011893*(In[-4]+In[4])
                              +   0.044728*(In[-3]+In[3])
                              +   0.115163*(In[-2]+In[2])
                              +   0.203084*(In[-1]+In[1])
                              +   0.245348*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num8_0_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_REAL4_Num9_1_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000002*(In[-9]+In[9])
                              +   0.000021*(In[-8]+In[8])
                              +   0.000180*(In[-7]+In[7])
                              +   0.001180*(In[-6]+In[6])
                              +   0.005790*(In[-5]+In[5])
                              +   0.021257*(In[-4]+In[4])
                              +   0.058442*(In[-3]+In[3])
                              +   0.120331*(In[-2]+In[2])
                              +   0.185587*(In[-1]+In[1])
                              +   0.214420*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num9_1_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-9),-9,9,15,false)       {
      }
};

class cConvolSpec_REAL4_Num10_2_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000000*(In[-11]+In[11])
                              +   0.000003*(In[-10]+In[10])
                              +   0.000025*(In[-9]+In[9])
                              +   0.000162*(In[-8]+In[8])
                              +   0.000848*(In[-7]+In[7])
                              +   0.003552*(In[-6]+In[6])
                              +   0.011932*(In[-5]+In[5])
                              +   0.032151*(In[-4]+In[4])
                              +   0.069497*(In[-3]+In[3])
                              +   0.120524*(In[-2]+In[2])
                              +   0.167697*(In[-1]+In[1])
                              +   0.187216*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num10_2_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-11),-11,11,15,false)       {
      }
};

class cConvolSpec_REAL4_Num11_3_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000001*(In[-12]+In[12])
                              +   0.000006*(In[-11]+In[11])
                              +   0.000037*(In[-10]+In[10])
                              +   0.000184*(In[-9]+In[9])
                              +   0.000764*(In[-8]+In[8])
                              +   0.002686*(In[-7]+In[7])
                              +   0.007989*(In[-6]+In[6])
                              +   0.020090*(In[-5]+In[5])
                              +   0.042721*(In[-4]+In[4])
                              +   0.076821*(In[-3]+In[3])
                              +   0.116815*(In[-2]+In[2])
                              +   0.150213*(In[-1]+In[1])
                              +   0.163347*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num11_3_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

class cConvolSpec_REAL4_Num12_4_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000003*(In[-13]+In[13])
                              +   0.000015*(In[-12]+In[12])
                              +   0.000064*(In[-11]+In[11])
                              +   0.000243*(In[-10]+In[10])
                              +   0.000815*(In[-9]+In[9])
                              +   0.002409*(In[-8]+In[8])
                              +   0.006268*(In[-7]+In[7])
                              +   0.014355*(In[-6]+In[6])
                              +   0.028944*(In[-5]+In[5])
                              +   0.051370*(In[-4]+In[4])
                              +   0.080260*(In[-3]+In[3])
                              +   0.110386*(In[-2]+In[2])
                              +   0.133648*(In[-1]+In[1])
                              +   0.142444*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num12_4_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-13),-13,13,15,false)       {
      }
};

class cConvolSpec_REAL4_Num13_5_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000002*(In[-15]+In[15])
                              +   0.000009*(In[-14]+In[14])
                              +   0.000035*(In[-13]+In[13])
                              +   0.000116*(In[-12]+In[12])
                              +   0.000354*(In[-11]+In[11])
                              +   0.000978*(In[-10]+In[10])
                              +   0.002456*(In[-9]+In[9])
                              +   0.005594*(In[-8]+In[8])
                              +   0.011569*(In[-7]+In[7])
                              +   0.021715*(In[-6]+In[6])
                              +   0.036995*(In[-5]+In[5])
                              +   0.057207*(In[-4]+In[4])
                              +   0.080296*(In[-3]+In[3])
                              +   0.102297*(In[-2]+In[2])
                              +   0.118295*(In[-1]+In[1])
                              +   0.124165*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num13_5_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-15),-15,15,15,false)       {
      }
};

class cConvolSpec_REAL4_Num14_6_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000003*(In[-17]+In[17])
                              +   0.000009*(In[-16]+In[16])
                              +   0.000028*(In[-15]+In[15])
                              +   0.000080*(In[-14]+In[14])
                              +   0.000216*(In[-13]+In[13])
                              +   0.000542*(In[-12]+In[12])
                              +   0.001263*(In[-11]+In[11])
                              +   0.002735*(In[-10]+In[10])
                              +   0.005501*(In[-9]+In[9])
                              +   0.010280*(In[-8]+In[8])
                              +   0.017847*(In[-7]+In[7])
                              +   0.028788*(In[-6]+In[6])
                              +   0.043143*(In[-5]+In[5])
                              +   0.060071*(In[-4]+In[4])
                              +   0.077709*(In[-3]+In[3])
                              +   0.093397*(In[-2]+In[2])
                              +   0.104291*(In[-1]+In[1])
                              +   0.108198*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num14_6_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-17),-17,17,15,false)       {
      }
};

class cConvolSpec_REAL4_Num15_7_5_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000001*(In[-20]+In[20])
                              +   0.000004*(In[-19]+In[19])
                              +   0.000011*(In[-18]+In[18])
                              +   0.000030*(In[-17]+In[17])
                              +   0.000074*(In[-16]+In[16])
                              +   0.000176*(In[-15]+In[15])
                              +   0.000396*(In[-14]+In[14])
                              +   0.000842*(In[-13]+In[13])
                              +   0.001693*(In[-12]+In[12])
                              +   0.003217*(In[-11]+In[11])
                              +   0.005781*(In[-10]+In[10])
                              +   0.009826*(In[-9]+In[9])
                              +   0.015793*(In[-8]+In[8])
                              +   0.024006*(In[-7]+In[7])
                              +   0.034507*(In[-6]+In[6])
                              +   0.046910*(In[-5]+In[5])
                              +   0.060307*(In[-4]+In[4])
                              +   0.073321*(In[-3]+In[3])
                              +   0.084303*(In[-2]+In[2])
                              +   0.091667*(In[-1]+In[1])
                              +   0.094262*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num15_7_5_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-20),-20,20,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num16_1_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num16_1_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-5),-5,5,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num17_2_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num17_2_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num18_3_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num18_3_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num19_4_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num19_4_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-10),-10,10,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num20_5_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num20_5_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-12),-12,12,15,false)       {
      }
};

class cConvolSpec_REAL4_Num16_1_3_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num16_1_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-5),-5,5,15,false)       {
      }
};

class cConvolSpec_REAL4_Num17_2_3_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num17_2_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-6),-6,6,15,false)       {
      }
};

class cConvolSpec_REAL4_Num18_3_3_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num18_3_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_REAL4_Num19_4_3_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num19_4_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

class cConvolSpec_REAL4_Num20_5_3_15 : public cConvolSpec<REAL4>
{
   public :
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

      cConvolSpec_REAL4_Num20_5_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num21_1_3_15 : public cConvolSpec<U_INT2>
{
   public :
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

      cConvolSpec_U_INT2_Num21_1_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-8),-8,8,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num22_2_3_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-11]+In[11])
                              +   3*(In[-10]+In[10])
                              +   10*(In[-9]+In[9])
                              +   38*(In[-8]+In[8])
                              +   120*(In[-7]+In[7])
                              +   325*(In[-6]+In[6])
                              +   755*(In[-5]+In[5])
                              +   1503*(In[-4]+In[4])
                              +   2569*(In[-3]+In[3])
                              +   3766*(In[-2]+In[2])
                              +   4737*(In[-1]+In[1])
                              +   5114*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num22_2_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-11),-11,11,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num23_4_3_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-16]+In[16])
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
                              +   3234*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num23_4_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-16),-16,16,15,false)       {
      }
};

class cConvolSpec_U_INT2_Num24_5_3_15 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                16384
                              +   1*(In[-21]+In[21])
                              +   1*(In[-20]+In[20])
                              +   3*(In[-19]+In[19])
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
                              +   747*(In[-8]+In[8])
                              +   997*(In[-7]+In[7])
                              +   1282*(In[-6]+In[6])
                              +   1585*(In[-5]+In[5])
                              +   1886*(In[-4]+In[4])
                              +   2160*(In[-3]+In[3])
                              +   2378*(In[-2]+In[2])
                              +   2520*(In[-1]+In[1])
                              +   2570*(In[0])
                           )>>15;
               In++;
          }
      }

      cConvolSpec_U_INT2_Num24_5_3_15(INT * aFilter):
           cConvolSpec<U_INT2>(aFilter-(-21),-21,21,15,false)       {
      }
};

class cConvolSpec_REAL4_Num21_1_3_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000001*(In[-10]+In[10])
                              +   0.000011*(In[-9]+In[9])
                              +   0.000087*(In[-8]+In[8])
                              +   0.000532*(In[-7]+In[7])
                              +   0.002551*(In[-6]+In[6])
                              +   0.009614*(In[-5]+In[5])
                              +   0.028462*(In[-4]+In[4])
                              +   0.066194*(In[-3]+In[3])
                              +   0.120951*(In[-2]+In[2])
                              +   0.173648*(In[-1]+In[1])
                              +   0.195894*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num21_1_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-10),-10,10,15,false)       {
      }
};

class cConvolSpec_REAL4_Num22_2_3_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000003*(In[-12]+In[12])
                              +   0.000015*(In[-11]+In[11])
                              +   0.000074*(In[-10]+In[10])
                              +   0.000317*(In[-9]+In[9])
                              +   0.001165*(In[-8]+In[8])
                              +   0.003671*(In[-7]+In[7])
                              +   0.009929*(In[-6]+In[6])
                              +   0.023041*(In[-5]+In[5])
                              +   0.045878*(In[-4]+In[4])
                              +   0.078385*(In[-3]+In[3])
                              +   0.114918*(In[-2]+In[2])
                              +   0.144571*(In[-1]+In[1])
                              +   0.156067*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num22_2_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-12),-12,12,15,false)       {
      }
};

class cConvolSpec_REAL4_Num23_4_3_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000002*(In[-19]+In[19])
                              +   0.000005*(In[-18]+In[18])
                              +   0.000014*(In[-17]+In[17])
                              +   0.000039*(In[-16]+In[16])
                              +   0.000101*(In[-15]+In[15])
                              +   0.000245*(In[-14]+In[14])
                              +   0.000560*(In[-13]+In[13])
                              +   0.001203*(In[-12]+In[12])
                              +   0.002433*(In[-11]+In[11])
                              +   0.004626*(In[-10]+In[10])
                              +   0.008275*(In[-9]+In[9])
                              +   0.013922*(In[-8]+In[8])
                              +   0.022032*(In[-7]+In[7])
                              +   0.032798*(In[-6]+In[6])
                              +   0.045924*(In[-5]+In[5])
                              +   0.060486*(In[-4]+In[4])
                              +   0.074936*(In[-3]+In[3])
                              +   0.087326*(In[-2]+In[2])
                              +   0.095723*(In[-1]+In[1])
                              +   0.098698*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num23_4_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-19),-19,19,15,false)       {
      }
};

class cConvolSpec_REAL4_Num24_5_3_15 : public cConvolSpec<REAL4>
{
   public :
      void Convol(REAL4 * Out,REAL4 * In,int aK0,int aK1)
      {
          In+=aK0;
          Out+=aK0;
          for (int aK=aK0; aK<aK1 ; aK++)
          {
               *(Out++) =  (
                                 0
                              +   0.000003*(In[-23]+In[23])
                              +   0.000007*(In[-22]+In[22])
                              +   0.000016*(In[-21]+In[21])
                              +   0.000035*(In[-20]+In[20])
                              +   0.000073*(In[-19]+In[19])
                              +   0.000150*(In[-18]+In[18])
                              +   0.000295*(In[-17]+In[17])
                              +   0.000558*(In[-16]+In[16])
                              +   0.001016*(In[-15]+In[15])
                              +   0.001779*(In[-14]+In[14])
                              +   0.002997*(In[-13]+In[13])
                              +   0.004858*(In[-12]+In[12])
                              +   0.007575*(In[-11]+In[11])
                              +   0.011364*(In[-10]+In[10])
                              +   0.016402*(In[-9]+In[9])
                              +   0.022778*(In[-8]+In[8])
                              +   0.030433*(In[-7]+In[7])
                              +   0.039119*(In[-6]+In[6])
                              +   0.048380*(In[-5]+In[5])
                              +   0.057565*(In[-4]+In[4])
                              +   0.065899*(In[-3]+In[3])
                              +   0.072581*(In[-2]+In[2])
                              +   0.076911*(In[-1]+In[1])
                              +   0.078411*(In[0])
                           );
               In++;
          }
      }

      cConvolSpec_REAL4_Num24_5_3_15(REAL8 * aFilter):
           cConvolSpec<REAL4>(aFilter-(-23),-23,23,15,false)       {
      }
};

}
