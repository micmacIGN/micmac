/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


/*
class cTest
{
   public :
     explicit cTest(int);
     explicit cTest(const cTest &);
};

void f()
{
    cTest i = cTest(3);
}
*/


#include "StdAfx.h"

/*
template <> Pt2d<double>::Pt2d(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
template <> Pt2d<float>::Pt2d(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
template <> Pt2d<int>::Pt2d(const Pt2d<double>& p) : x (round_ni(p.x)), y (round_ni(p.y)) {};
*/


#include <cstring>

long int lPowi(int aN,int anExp)
{
    long int aRes = 1;
    for (int anE=0 ; anE<anExp ; anE++)
        aRes *= aN;
    return aRes;
}

cDecimal::cDecimal(int aMant,int aPow) :
   mMant (aMant),
   mExp  (aPow)
{
}

double cDecimal::RVal() const
{
   return mMant * pow(10.0,mExp);
}


long int cDecimal::Mul10() const
{
    return lPowi(10,mExp);
}
long int cDecimal::Div10() const
{
    return lPowi(10,-mExp);
}

const int & cDecimal::Exp() const
{
   return mExp;
}

const int & cDecimal::Mant() const
{
   return mMant;
}

double cDecimal::Arrondi(double aV0) const
{
    long int aD10 =  Div10();
    long double aV = aV0;
    aV *= aD10;
    long int aLMant = Mul10() * (long int) mMant;
    return (lround_ni(aV/aLMant)/(long double)aD10)*(long double)aLMant;
}


//==============================================================
//==============================================================
//==============================================================

#define SizeTabRound 25
int TabRound[SizeTabRound] = {10,11,12,14,15,16,18,20,22,25,28,30,32,35,40,45,50,55,60,65,70,75,80,90,100};

cDecimal StdRound(const double & aD,int aNbDigit,int * aTabR,int aSizeR)
{
   int aL10 = round_down(log10(aD)) - aNbDigit+1;

   double aP10 = pow(10.0,aL10);
   double aVI = aD/aP10;

   int aBestK = -1;
   double aDifMin=1e20;
   for (int aK=0 ; aK<aSizeR ; aK++)
   {
       double aDif = ElAbs(aVI-aTabR[aK]);
       if (aDif < aDifMin)
       {
           aDifMin=aDif;
           aBestK = aTabR[aK];
       }
   }
   return  cDecimal(aBestK,aL10);
   // aBestK * aP10;
}

cDecimal StdRound(const double & aD)
{
   return StdRound(aD,2,TabRound,SizeTabRound);
}



FBool::FBool(U_INT1 aVl) :
    mVal (aVl)
{
}



const FBool FBool::True(3);
const FBool FBool::MayBe(2);
const FBool FBool::False(1);

bool MSBF_PROCESSOR()
{
    static bool init = false;
    static bool res  = true; // bidon

    if (! init)
    {
        U_INT2 ui2=0;
        char * c = (char *) &ui2;

        c[0] = 0;
        c[1] = 1;


        res  = (ui2 == 1);
        init = true;
   }
   return res;
}


void to_lsb_rep_2(void * adr)
{
   if (MSBF_PROCESSOR())
      byte_inv_2(adr);
}
void to_lsb_rep_4(void * adr)
{
   if (MSBF_PROCESSOR())
      byte_inv_4(adr);
}

void to_msb_rep_2(void * adr)
{
   if (!MSBF_PROCESSOR())
      byte_inv_2(adr);
}
void to_msb_rep_4(void * adr)
{
   if (!MSBF_PROCESSOR())
      byte_inv_4(adr);
}







void mem_raz(void * adr,tFileOffset nb)
{
    memset(adr,0,nb.CKK_AbsLLO());
}

void  set_fonc_x(INT * res,INT x0,INT x1)
{
    while (x0 < x1)
          *(res++) = x0++;
}


int index_values_complex_nul(const REAL * x,const REAL *y,INT nb)
{
     for (int i=0; i<nb ; i++)
        if ((x[i] == 0) && (y[i] == 0))
           return i;

    return INDEX_NOT_FOUND;
}

int index_values_strict_neg(const REAL * t,INT nb)
{
     for (int i=0; i<nb ; i++)
        if (t[i] < 0)
           return i;

    return INDEX_NOT_FOUND;
}


int index_values_neg_or_null(const REAL * t,INT nb)
{
     for (int i=0; i<nb ; i++)
        if (t[i] <= 0)
           return i;

    return INDEX_NOT_FOUND;
}


int index_values_not_acos(const REAL * t,INT nb)
{
     for (int i=0; i<nb ; i++)
        if(  (t[i]<-1.0)  ||  (t[i]>1.0)  )
           return i;

    return INDEX_NOT_FOUND;
}


void tab_not_log(INT * out,const INT * in,INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = (! in[i]);
}

void tab_not_bit_by_bit(INT * out,const INT * in,INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = (~ in[i]);
}





void round_up (INT * out  ,const REAL * in, INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = round_up(in[i]);
}

void round_down (INT * out  ,const REAL * in, INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = round_down(in[i]);
}


void round_ni (INT * out  ,const REAL * tab_in, INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = round_ni(tab_in[i]);
}


void round_ni_inf (INT * out  ,const REAL * in, INT nb)
{
     for (int i =0; i<nb ; i++)
         out[i] = round_ni_inf(in[i]);
}




Interval::Interval(REAL v0,REAL v1) : _v0 (ElMin(v0,v1)), _v1 (ElMax(v0,v1)){}
Interval::Interval() : _v0 (0), _v1(0) {}

REAL Interval::dist(const Interval & I2)
{
     return ElMax
            (
                0.0,
                  ElMax(_v0,I2._v0)
                - ElMin(_v1,I2._v1)
            );
}


#if (0)

INT round_up(INT a,INT b)
{
   return ((a+b-1)/b)*b;
}


// return the smallest integral value >= r
INT round_up(REAL r)
{
       INT i = (INT) r;
       return i + (i < r);
}

// return the smallest integral value > r
INT round_Uup(REAL r)
{
       INT i = (INT) r;
       return i + (i <= r);
}


// return the highest integral value <= r
INT round_down(REAL r)
{
       INT i = (INT) r;
       return i - (i > r);
}

// return the highest integral value < r
INT round_Ddown(REAL r)
{
       INT i = (INT) r;
       return i - (i >= r);
}



// return the integral value closest to r
// if r = i +0.5 (i integer) return i+1
INT round_ni(REAL r)
{
       INT i = (INT) r;
       i -= (i > r);
       return i+ ((i+0.5) <= r) ;
}

// return the integral value closest to r
// if r = i +0.5 (i integer) return i
INT round_ni_inf(REAL r)
{
       INT i = (INT) r;
       i -= (i > r);
       return i+ ((i+0.5) < r) ;
}

// return the real division of a by b; eq
// complies with the mathematical property
//     b*r <= a < b * (r+1)
//   Unpredictable for b < 0.

INT Elise_div(INT a,INT b)
{
       INT res = a / b;
       return res - (res * b > a);
}


// work only when b > 0
INT mod(INT a,INT b)
{
    INT r = a%b;
    return (r <0) ? (r+b) : r;
}

INT mod256(INT a)
{
    INT r = a%256;
    return (r <0) ? (r+256) : r;
}


// work only also when b < 0
INT mod_gen(INT a,INT b)
{

    INT r = a%b;
    return (r <0) ? (r+ ((b>0) ? b : -b)) : r;
}

REAL square(REAL v1) {return (v1*v1);}

INT sub_bit(INT v,INT k0,INT k1)
        { return (v >> k0) & ((1 << (k1-k0)) -1);}


INT set_sub_bit(INT v,INT new_v,INT k0,INT k1)
{
      INT masq = (1 << (k1-k0)) -1;

      return
                (v & ~(masq << k0))  // efface les bits entre k0 et k1
              | ((new_v&masq) << k0);
}


INT kth_bit(INT v,INT k)             { return (v & (1<<k)) != 0 ; }

INT kth_bit_to_1(INT v, INT k)       { return v | (1<< k)       ; }

INT kth_bit_to_0(INT v, INT k)       { return v & (~ (1<< k))   ; }

INT set_kth_bit_to_1(INT & v, INT k) { return v |=  1<< k       ; }

INT set_kth_bit_to_0(INT & v, INT k) { return v &=  (~ (1<< k)) ; }

INT nb_bits_to_nb_byte(INT nb_bits) { return (nb_bits + 7) / 8; }

INT kth_bit(const U_INT1* v,INT k)
    { return (v[k/8]&(1<<(k%8)))!= 0;}

INT kth_bit_msbf(const U_INT1* v,INT k)
    { return (v[k/8]&(1<<(7-k%8)))!= 0;}

U_INT1 kth_bit_to_1(const U_INT1 * v, INT k)
       { return v[k/8] | (1<< (k%8)) ;}

#endif // ! CPP_OPTIMIZE

INT  inv_bits_order(INT val,INT nbb)
{
     INT res = 0;
     for (INT i=0; i<nbb; i++)
         if (kth_bit(val,i))
            res = set_kth_bit_to_1(res,nbb-i-1);
     return res;
}


double arrondi_inf(double aVal,double aPer)
{
   return ((long double)aPer) * lround_down(aVal/aPer);
}
double arrondi_sup(double aVal,double aPer)
{
   return ((long double)aPer) * lround_up(aVal/aPer);
}

double arrondi_ni(double aVal,double aPer)
{
   return ((long double)aPer) * lround_ni(aVal/aPer);
}


Pt2dr arrondi_ni(const Pt2dr & aP,double aPer)
{
    return Pt2dr(arrondi_ni(aP.x,aPer),arrondi_ni(aP.y,aPer));
}


REAL Pow(REAL x,INT i)
{
    if (i>0)
       return   (i&1) ? (x * ElSquare(Pow(x,i/2))) :  ElSquare(Pow(x,i/2));

     if (i<0)
        return 1.0 / Pow(x,-i);

     return   1.0;
}

INT Pow_of_2_sup(INT x)
{
    INT i;
    for (i = 1 ; i < x; i *= 2);
    return i;
}

bool is_pow_of_2(INT x)
{
    if (x <=0)
       return false;

    INT l2 = round_ni(log(double(x))/log(double(2.0)));
    return x == (1<<l2);
}

int NbBitsOfFlag(int aFlag)
{
   int aRes=0;
   for (int aK=0 ;aK<30 ; aK++)
     if (aFlag &(1<<aK))
        aRes++;

   return  aRes;
}

/*
     Algorithme  tire de "Introduction a la theorie des nombres"
      (Jean Marie De Koninck, edition Modulo, pp 146--160):

     - On appelle fraction continue l'expression [a1,a2,a3,...] definie
       par :

                                        1
       [a1,a2,a3,...] =  a1+ -------------------------
                              a2 +      1
                                  --------------------
                                   a3+  1
                                      ----------------
                                       ...
       On demontre que la fraction rationelle correspondant au developpement
       de n premier terme de [a1,a2,a3,...] s'obtient par (j'ai rajoute
       les termes p-1,q-1 qui facilite l'implementation):

        p[-1] = 0 ; p[0] = 1 ; p[1] = a1 ;
        q[-1] = 1 ; q[0] = 0 ; q[1] = 1  ;

        et

         p[n] =  a[n] p[n-1] + p[n-2]
         q[n] =  a[n] q[n-1] + q[n-2]


        Soit maintenant un nombre reel r, on demontre que son developpement
        en fraction continue s'obtient par la recurence suivante :

         r1 = r
                         1
         r[n+1] = -------------
                  E(r[n]) - r[n]

          a[n] = E(r[n]);

        Le programme qui suit n'est qu'une implementation directe de
        ce qui precede.

*/


void rationnal_approx_std(REAL r,INT & p,INT &q)
{
     REAL p_M2 = 0;
     REAL q_M2 = 1;

     REAL p_M1 = 1;
     REAL q_M1 = 0;


     while(1)
     {
          REAL a = floor(r);

          REAL  p0 = a * p_M1 + p_M2;
          REAL  q0 = a * q_M1 + q_M2;

          if ((p0 > 1e9) || (q0 > 1e9))
          {
                 p = (INT) p_M1;
                 q = (INT) q_M1;
                 return;
          }

          r = r-a;
          if (r < 1e-9)
          {
                 p = (INT) p0;
                 q = (INT) q0;
                 return;
          }
          r = 1/r;


          p_M2 = p_M1;
          q_M2 = q_M1;

          p_M1 = p0;
          q_M1 = q0;
     }
}


void rationnal_approx(REAL r,INT & p,INT &q)
{
    INT sign = (r >= 0) ? 1 : -1;
    r *= sign;

    rationnal_approx_std(r,p,q);

    p *= sign;
}



Pt2dr rto_user_geom(Pt2dr p,Pt2dr t,Pt2dr s)
{
    return Pt2dr
           (
                rto_user_geom(p.x,t.x,s.x),
                rto_user_geom(p.y,t.y,s.y)
           );
}

/*******************************************/
/*                                         */
/*         cINT8ImplemSetInt               */
/*                                         */
/*******************************************/

cINT8ImplemSetInt::cINT8ImplemSetInt() :
   mFlag (0)
{
}


void cINT8ImplemSetInt::Add(int anInt)
{
  mFlag |= ((INTByte8)1) << (INTByte8)anInt;
}


bool cINT8ImplemSetInt::IsIn(int anInt) const
{
   return (mFlag &(((INTByte8)1)  << (INTByte8)anInt)) != 0;
}


int  cINT8ImplemSetInt::Capacite()
{
   return 62;
}

bool cINT8ImplemSetInt::operator < (const cINT8ImplemSetInt & aS2) const
{
   return mFlag < aS2.mFlag;
}

int  cINT8ImplemSetInt::NumOrdre(int aI,bool svp) const
{
   if (! svp)
      ELISE_ASSERT(IsIn(aI),"Inc in cINT8ImplemSetInt::NumOrdre");
   int aRes = 0;
   for (int aK=0; aK<aI ; aK++)
       if (IsIn(aK))  
           aRes++;
   return aRes;
}
int  cINT8ImplemSetInt::NumOrdre(int aI) const
{
   return NumOrdre(aI,false);
}


/*******************************************/
/*                                         */
/*             cVarSetIntMultiple          */
/*                                         */
/*******************************************/

/*
class cVarSetIntMultiple
{
    public :
       cVarSetIntMultiple();
       void Add(int anInt);
       bool IsIn (int anInt) const;
       int  NumOrdre(int aI) const;  // Nombre d'Entier < a aI

       bool operator < (const cVarSetIntMultiple &) const;
    private :
        Pt2d<INT>  NumSub(const int & anI) const; // x le set, y le I dans le set
        mutable std::vector<cINT8ImplemSetInt>  mSets;
};
*/

cVarSetIntMultiple::cVarSetIntMultiple()
{
}

Pt2di cVarSetIntMultiple::NumSub(const int & anI) const
{
   int x= anI/cINT8ImplemSetInt::Capacite();
   int y= anI%cINT8ImplemSetInt::Capacite();

   while (x>=int(mSets.size()))
   {
        mSets.push_back(cINT8ImplemSetInt());
   }
   return Pt2di(x,y);
}


void cVarSetIntMultiple::Add(int  anI) 
{
   Pt2di aP = NumSub(anI);
   mSets[aP.x].Add(aP.y);
}

bool cVarSetIntMultiple::IsIn(int anI) const
{
   Pt2di aP = NumSub(anI);
   return mSets[aP.x].IsIn(aP.y);
}

int  cVarSetIntMultiple::NumOrdre(int aI) const
{
   int aRes =0;
   Pt2di aP = NumSub(aI);
   for (int aKx=0 ; aKx<aP.x ; aKx++)
       aRes += mSets[aKx].NumOrdre(cINT8ImplemSetInt::Capacite(),true);

   return aRes + mSets[aP.x].NumOrdre(aP.y);
}


bool cVarSetIntMultiple::operator < (const cVarSetIntMultiple & aS2) const
{
   int aNb1 = (int)mSets.size();
   int aNb2 = (int)aS2.mSets.size();

   if (aNb1 < aNb2) return true;
   if (aNb2 < aNb1) return false;

   for (int aK=0 ; aK<aNb1 ; aK++)
   {
       if (mSets[aK] < aS2.mSets[aK]) return true;
       if (aS2.mSets[aK] < mSets[aK]) return false;
   }
   return false;
}

int cVarSetIntMultiple::Capacite() const
{
    return (int)(mSets.size() * cINT8ImplemSetInt::Capacite());
}

/*
*/


/*******************************************/
/*                                         */
/*             cSetIntMultiple             */
/*                                         */
/*******************************************/


template <const int NbI> 
   cSetIntMultiple<NbI>::cSetIntMultiple()
{
}


template <const int NbI> 
  Pt2di cSetIntMultiple<NbI>::NumSub(const int & anI) const
{
   int x= anI/cINT8ImplemSetInt::Capacite();
   int y= anI%cINT8ImplemSetInt::Capacite();

   ELISE_ASSERT(x<NbI,"Over Capa in cSetIntMultiple");
   return Pt2di(x,y);
}

template <const int NbI> 
  void cSetIntMultiple<NbI>::Add(int  anI) 
{
   Pt2di aP = NumSub(anI);
   mSets[aP.x].Add(aP.y);
}

template <const int NbI>
  bool cSetIntMultiple<NbI>::IsIn(int anI) const
{
   Pt2di aP = NumSub(anI);
   return mSets[aP.x].IsIn(aP.y);
}

template <const int NbI>
int  cSetIntMultiple<NbI>::NumOrdre(int aI) const
{
   int aRes =0;
   Pt2di aP = NumSub(aI);
   for (int aKx=0 ; aKx<aP.x ; aKx++)
       aRes += mSets[aKx].NumOrdre(cINT8ImplemSetInt::Capacite(),true);

   return aRes + mSets[aP.x].NumOrdre(aP.y);
}


template <const int NbI> 
   int cSetIntMultiple<NbI>::Capacite()
{
    return NbI * cINT8ImplemSetInt::Capacite();
}


template <const int NbI> 
  bool cSetIntMultiple<NbI>::operator < (const cSetIntMultiple<NbI> & aS2) const
{
   for (int aK=0 ; aK<NbI ; aK++)
   {
       if (mSets[aK] < aS2.mSets[aK]) return true;
       if (aS2.mSets[aK] < mSets[aK]) return false;
   }
   return false;
}

template class cSetIntMultiple<2>;
template class cSetIntMultiple<3>;
template class cSetIntMultiple<4>;
template class cSetIntMultiple<5>;
template class cSetIntMultiple<6>;


/*******************************************/
/*                                         */
/*             ::                          */
/*                                         */
/*******************************************/

     //  PRECOND EN  2 SIN (ATAN /2 )

double Dl_f2SAtRxS2SRx(double x)
{
    return 1 - (1/3.0 + 1/24.0) * x + (1/5.0 +1/24.0) *x*x;
}
double Std_f2SAtRxS2SRx(double x)
{
   x  = sqrt(ElAbs(x));
   return (2* sin(atan(x)/2) ) /x;
}
double  f2SAtRxS2SRx(double x)
{
   x = ElAbs(x);
   if  (x<1e-5) return Dl_f2SAtRxS2SRx(x);
   return Std_f2SAtRxS2SRx(x);
}




double Dl_Der2SAtRxS2SRx(double x)
{
   return - (1/3.0 + 1/24.0) + 2 * x *(1/5.0 +1/24.0);
}
double Std_Dl_Der2SAtRxS2SRx(double x)
{
   double aSqrX = sqrt(x);
   double aAtS2 = atan(aSqrX)/2.0;


   return (cos(aAtS2)/(1+x) -2*sin(aAtS2)/aSqrX) / (2*x);
}
double Der2SAtRxS2SRx(double x)
{
   x = ElAbs(x);
   if (x<1e-5) return Dl_Der2SAtRxS2SRx(x);

    return Std_Dl_Der2SAtRxS2SRx(x);
}

double f4S2AtRxS2(double x)
{
   return  ElSquare(2* sin(  atan(sqrt(ElAbs(x)))  /2.0));
}

double  Dl_Der4S2AtRxS2(double x)
{
   return (1-x/2.0) / (1+x);
}
double Std_Der4S2AtRxS2(double x)
{
   double aSqX = sqrt(x);
   return (sin(atan(aSqX))/aSqX) / (1+x);
}
double Der4S2AtRxS2(double x)
{
   x = ElAbs(x);
   if (x<1e-5) return Dl_Der4S2AtRxS2(x);
   return Std_Der4S2AtRxS2(x);
}

double Dl_Tg2AsRxS2SRx(double x)
{
   return 1 + x *(1/24.0+1/3.0);
}
double Std_Tg2AsRxS2SRx(double x)
{
   double aSqX = sqrt(x);
   return tan(2*asin(aSqX/2.0)) / aSqX;
}
double Tg2AsRxS2SRx(double x)
{
   if (x<1e-5) return Dl_Tg2AsRxS2SRx(x);
   return Std_Tg2AsRxS2SRx(x);
}
Fonc_Num Tg2AsRxS2SRx(Fonc_Num aF)
{
   ELISE_ASSERT(false,"Tg2AsRxS2SRx(Fonc_Num aF)");
   return 0;
}

//  Operateur utile au devt en four des fonction radiale

double CosRx(double anX)
{
   return cos(sqrt(ElAbs(anX)));
}

double SinCardRx(double anX)
{
   anX =  ElAbs(anX);
   if (anX <1e-5) return 1 - anX/6.0 + (anX*anX)/120.0;

   anX = sqrt(anX);
   return sin(anX) / anX;
}

     //  PRECOND EN ATAN
double AtRxSRx(double x)
{
   x = ElAbs(x);
   if  (x<1e-5) return (1-x/3.0+(x*x)/5.0) ;    // dev limite
   x  = sqrt(x);
   return atan(x)/x;
}

double TgRxSRx(double x)
{
   x = ElAbs(x);
   if  (x<1e-6) return (1+x/3.0) ;    // dev limite
   x  = sqrt(x);
   return tan(x)/x;
}
Fonc_Num TgRxSRx(Fonc_Num aF)
{
   ELISE_ASSERT(false,"TgRxSRx(Fonc_Num aF)");
   return 0;
}



double Square(double x){return x*x;}

double DerAtRxSRx(double x)
{
    x = ElAbs(x);

    if (x<1e-5) 
       return (-1/3.0 + (2.0/5.0)*x -(3.0/7.0)*x);

    return (1/(1+x)-AtRxSRx(x))/(2*x); 
}

double At2Rx(double x)
{
   return ElSquare(atan(sqrt(ElAbs(x))));
}

double DerAt2Rx(double x)
{
   return AtRxSRx(x) / (1+ElAbs(x));
}

 REAL angle_mod_real(REAL a,REAL b)
{
     REAL res = mod_real(a,b);
     if (res > ElAbs(b/2))
        res -= ElAbs(b);
     return res;
}


double IsInf(double aV) {return std_isinf(aV);}
double IsNan(double aV)    {return std_isnan(aV);}
double IsBadNum(double aV) {return std_isinf(aV) || std_isnan(aV);}



/*
    Soit Z dans l'intervalle ouvert I1 [aZ1Min,aZ1Max[,
    on recherche dans l'intervalle ouvert I0 [aZ0Min,aZ0Max[,
    un intervalle ferme, non vide, le plus proche possible
    [aZ+aDzMin,aZ+aDzMax].

    De plus ce calcul doit generer des connexion symetrique.

    Ex  :
        I1 = [10,30[
        I0 = [5,20[

        MaxDeltaZ = 2


        Z = 13 ->    Delta = [-2,2]   // aucune contrainte
        Z = 18 ->    Delta = [-2,1]   // Pour que ca reste dans I0
        Z = 25 ->    Delta = [-6,-6]  //  Pour que l'intersection soit non vide avec I0
        Z = 10 ->    Delta = [-5,-1]  // principe de symetrie, dans l'autre sens                                      // les points [5,9] de I0 devront etre connecte a 10

*/

void ComputeIntervaleDelta
              (
                  INT & aDzMin,
                  INT & aDzMax,
                  INT aZ,
                  INT MaxDeltaZ,
                  INT aZ1Min,
                  INT aZ1Max,
                  INT aZ0Min,
                  INT aZ0Max
              )
{
      aDzMin =   aZ0Min-aZ;
      if (aZ != aZ1Min)
         ElSetMax(aDzMin,-MaxDeltaZ);

      aDzMax = aZ0Max-1-aZ;
      if (aZ != aZ1Max-1)
         ElSetMin(aDzMax,MaxDeltaZ);

       // Si les intervalles sont vides, on relie
       // les bornes des intervalles a tous les points
       if (aDzMin > aDzMax)
       {
          if (aDzMax <0)
             aDzMin = aDzMax;
          else
             aDzMax = aDzMin;
       }
}

void BasicComputeIntervaleDelta
              (
                  INT & aDzMin,
                  INT & aDzMax,
                  INT aZ,
                  INT MaxDeltaZ,
                  INT aZ0Min,
                  INT aZ0Max
              )
{
   aDzMin = ElMax(-MaxDeltaZ,aZ0Min-aZ);
   aDzMax = ElMin(MaxDeltaZ,aZ0Max-1-aZ);
}

double FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec)
{
   double aRes = exp(- (sqrt(6*mCurNbIterFenSpec))/(1+aSzW));
   // std::cout << "ANCIEN " << Old_FromSzW2FactExp(aSzW,mCurNbIterFenSpec) << "\n";
   // std::cout << "FromSzW2FactExp : " << aRes << "\n"; getchar();
   return aRes;
}

double MoyHarmonik(const double & aV1,const double & aV2)
{
    return  1.0 /  (  ((1.0/aV1) + (1.0/aV2)) / 2.0) ;
}

double MoyHarmonik(const double & aV1,const double & aV2,const double & aV3)
{
    return  1.0 /  (  ((1.0/aV1) + (1.0/aV2) + (1.0/aV3)) / 2.0) ;
}



bool CmpPtsX(const Pt2df & aP1,const Pt2df & aP2) {return aP1.x < aP2.x;}

double PropPond(std::vector<Pt2df> &  aV,double aProp,int * aKMed)
{
/*
if (MPD_MM())
{
std::cout << "MedianPondMedianPond " << aV.size() << " " << aKMed << "\n";
}
*/
     ELISE_ASSERT(aV.size() !=0,"MedianPond with empty vector");
     std::sort(aV.begin(),aV.end(),CmpPtsX);
     double aSomP = 0;
     for (int aK=0 ; aK<int(aV.size()) ; aK++)
     {
          aSomP += aV[aK].y;
     }
     aSomP *= aProp;

     int aK=0;
     for ( ; (aK<int(aV.size()-1)) && (aSomP>0)  ; aK++)
     {
          aSomP -= aV[aK].y;
     }

     if (aKMed) 
     {
        *aKMed = aK;
     }

     return aV.at(aK).x;
}


double MedianPond(std::vector<Pt2df> &  aV,int * aKMed)
{
    return PropPond(aV,0.5,aKMed);
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
