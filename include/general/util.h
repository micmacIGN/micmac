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



#ifndef _ELISE_UTIL_H
#define _ELISE_UTIL_H

class cParamCalcVarUnkEl;
extern cParamCalcVarUnkEl * NullPCVU;
class tFileOffset;

extern const  std::string  TheFileMMDIR;  // MicMacInstalDir
void AnalyseContextCom(int argc,char ** argv);
void MMD_InitArgcArgv(int argc,char ** argv,int aNbArgMin=-1);
int NbProcSys();

extern void mem_raz(void *,tFileOffset);

extern int MemoArgc;
extern char ** MemoArgv;
extern std::string SubstArgcArvGlob(int aKSubst,std::string aSubst, bool aProtect = false);


std::string GetUnikId();
std::string Dir2Write(const std::string  DirChantier = "./" );

void ElExit(int aLine,const char * aFile,int aCode,const std::string & aMessage);
#define ElEXIT(aCode,aMessage) ElExit(__LINE__,__FILE__,aCode,aMessage)
//  Il existe des exit qui n'ont pas besoin d'etres traces, par exemple sur les help
#define StdEXIT(aCode)  exit(aCode)

void AddMessErrContext(const std::string & aMes);

int mm_getpid();

#define MEM_RAZ(x,nb) mem_raz((void *)(x),(nb)*sizeof(*(x)))


void test();
// constantly redefined to perform some test;

extern double Delay_Brkp;
void SleepProcess(double);
#define BRKP \
{\
    cout << "BRKP at " << __LINE__ << " of " << __FILE__ << "\n";\
    if (Delay_Brkp>0)\
        getchar();\
}

/*

    This file contains miscellaneous utilitaries
    (class and functions).

*/

const INT INDEX_NOT_FOUND = -1;

// FBoolA = "Fuzzy boolean" = boolean + option Fundef (ie undefined)

class FBool
{
    public :
       static const FBool True;
       static const FBool MayBe;
       static const FBool False;

      FBool operator || (const FBool & F2)  const
      {
         return FBool((mVal > F2.mVal) ? mVal : F2.mVal);
      }
      FBool operator && (const FBool & F2)  const
      {
         return FBool((mVal < F2.mVal) ? mVal : F2.mVal);
      }
      bool operator == (const FBool & F2)  const
      {
         return mVal == F2.mVal;
      }
      bool operator != (const FBool & F2)  const
      {
         return mVal != F2.mVal;
      }

      bool  BoolCast()
      {
           if ((*this)==True) return true;
           if ((*this)==False) return false;

           ELISE_ASSERT(false,"FBool::BoolCast , val is MayBe");
           return false;
      }




    private :
       FBool(U_INT1);
       U_INT1    mVal;
};


template <class Type> class Pt2d;


// Pour contourner le warning  [-Werror=unused-but-set-variable] de Gcc 4.6  FakeUseIt
template <class Type> void GccUse(const Type & ) {}



/*************************************************************/
/* template &        : ElMax,ElMin, ElSwap                   */
/*************************************************************/

/* Round to nearest integer. round_ni(0.5) = 1.
*/


// return the smallest integral value >= r
template<class Type> inline Type Tpl_round_up(REAL r)
{
       Type i = (Type) r;
       return i + (i < r);
}
inline INT round_up(REAL r) { return Tpl_round_up<int>(r); }
inline long int lround_up(REAL r) { return Tpl_round_up<long int>(r); }


// return the smallest integral value > r
template<class Type> inline Type Tpl_round_Uup(REAL r)
{
       Type i = (Type) r;
       return i + (i <= r);
}
inline INT round_Uup(REAL r) { return Tpl_round_Uup<int>(r); }


// return the highest integral value <= r
template<class Type> inline Type Tpl_round_down(REAL r)
{
       Type i = (Type) r;
       return i - (i > r);
}
inline INT round_down(REAL r) { return Tpl_round_down<int>(r); }
inline long int lround_down(REAL r) { return Tpl_round_down<long int>(r); }

// return the highest integral value < r
template<class Type> inline Type Tpl_round_Ddown(REAL r)
{
       Type i = (Type) r;
       return i - (i >= r);
}
inline INT round_Ddown(REAL r) { return Tpl_round_Ddown<int>(r); }



// return the integral value closest to r
// if r = i +0.5 (i integer) return i+1
template<class Type> inline Type Tpl_round_ni(REAL r)
{
       Type i = (Type) r;
       i -= (i > r);
       return i+ ((i+0.5) <= r) ;
}

inline INT round_ni(REAL r) { return Tpl_round_ni<int>(r); }
inline long int lround_ni(REAL r) { return Tpl_round_ni<long int>(r); }
/*
inline INTByte8 ll_round_ni(REAL r)
{
       INTByte8 i = (INTByte8) r;
       i -= (i > r);
       return i+ ((i+0.5) <= r) ;
}
*/





// return the integral value closest to r
// if r = i +0.5 (i integer) return i
inline INT round_ni_inf(REAL r)
{
       INT i = (INT) r;
       i -= (i > r);
       return i+ ((i+0.5) < r) ;
}

inline REAL signed_frac(REAL r)
{
    r = r -INT(r);
    if (r<-0.5) r++;
    if (r>0.5) r--;
    return r;
}

// return the real division of a by b; eq
// complies with the mathematical property
//     b*r <= a < b * (r+1)
//   Unpredictable for b < 0.

inline INT Elise_div(INT a,INT b)
{
       INT res = a / b;
       return res - ((res * b) > a);
}

// work only when b > 0

inline INT mod(INT a,INT b)
{
    INT r = a%b;
    return (r <0) ? (r+b) : r;
}

inline INT round_up(INT a,INT b)
{
   return ((a+b-1)/b)*b;
}

#if ((-1 & 255) == 255)
inline INT mod256(INT a) { return a & 255; }
#else
inline INT mod256(INT a) { return mod(a,256);}
#endif

// work only also when b < 0
inline INT mod_gen(INT a,INT b)
{

    INT r = a%b;
    return (r <0) ? (r+ ((b>0) ? b : -b)) : r;
}


inline INT arrondi_inf(INT a,INT b)
{
   return (a/b)*b;
}

inline INT arrondi_ni(INT a,INT b)
{
   return ((a+b/2)/b)*b;
}

inline INT arrondi_sup(INT a,INT b)
{
   return ((a+b-1)/b)*b;
}

 

double arrondi_inf(double aVal,double aPer);
double arrondi_sup(double aVal,double aPer);
double arrondi_ni(double aVal,double aPer);

inline REAL mod_real(REAL a,REAL b)
{
   REAL res =  a - b *round_down(a/b);
   while (res>=b) res -= b;
   while (res<0) res += b;
   return res;
}

inline REAL Centered_mod_real(REAL a,REAL b)
{
    REAL aRes = mod_real(a,b);
    if (aRes > (b/2)) aRes -= b;
    return aRes;
}


class cDecimal
{
    public :
        cDecimal(int aMant,int aPow);
        double RVal() const;
        const int &    Mant() const;
        const int &   Exp() const;
        double Arrondi(double aV) const;
      // T.Q RVAl = mMant * Mul10() / Div10()
        long int Mul10() const;
        long int Div10() const;
    public :
        int mMant;
        int mExp;
};



cDecimal StdRound(const double & aD,int aNbDigit,int * aTabR,int aSizeR);
cDecimal StdRound(const double & aD);


REAL angle_mod_real(REAL a,REAL b);


REAL Pow(REAL,INT);
INT Pow_of_2_sup(INT);
extern bool is_pow_of_2(INT );
int NbBitsOfFlag(int aFlag);

REAL El_logDeux(REAL);



template <class Type> void binarise(Type *,Type,INT);
template <class Type> void binarise(Type *,const Type *,Type,INT);
template <class Type> void neg_binarise(Type *,const Type *,Type,INT);


template <class Type> void set_fonc_id(Type *,Type v0,INT);

template <class Type> bool values_positive (const Type *t,INT nb);
template <class Type> bool values_positive_strict (const Type *t,INT nb);

template <class Type> bool values_in_range
           (const Type *t,INT nb,Type v_min,Type v_max);

template <class Type> INT index_values_out_of_range
             (const Type *t,INT nb,Type v_min,Type v_max);

template <class Type> INT index_vmax (const Type *t,INT nb);
template <class Type> INT index_vmin (const Type *t,INT nb);



template <class Type> bool values_all_inf_to
           (const Type *t,INT nb,Type v_max);

template <class Type> void proj_in_seg
                            (
                                Type *,
                                const Type *,
                                Type v_min,
                                Type v_max,
                                INT
                            );

template <class Tout,class Tin> void convert(Tout *,const Tin *,INT);


template <class Type> void set_cste(Type *,Type,INT);




template <class Type> Type * dup(const Type *,INT);
char * dup(const char *);
char * cat(const char * ch1,const char * ch2);



template <class Type> void auto_reverse_tab (Type *,INT nb);

template <class Type> void rotate_plus_data(Type *,INT i0,INT i1);
template <class Type> void rotate_moins_data(Type *,INT i0,INT i1);


// return INDEX_NOT_FOUND if all not null :

template <class Type> int index_values_null(const Type * t,INT nb);

template <class Type> void tab_Abs (Type * out,const Type * in,INT nb);

template <class Type> void tab_minus1 (Type * out,const Type * in,INT nb);

template <class Type> void tab_square (Type * out,const Type * in,INT nb);
template <class Type> void tab_cube (Type * out,const Type * in,INT nb);
template <class Type> void tab_pow4 (Type * out,const Type * in,INT nb);
template <class Type> void tab_pow5 (Type * out,const Type * in,INT nb);
template <class Type> void tab_pow6 (Type * out,const Type * in,INT nb);
template <class Type> void tab_pow7 (Type * out,const Type * in,INT nb);


void tab_not_log(INT * out,const INT * in,INT nb);
void tab_not_bit_by_bit(INT * out,const INT * in,INT nb);

void round_up (INT * out  ,const REAL * in, INT nb);
void round_down (INT * out  ,const REAL * in, INT nb);
void round_ni (INT * out  ,const REAL * in, INT nb);
void round_ni_inf (INT * out  ,const REAL * in, INT nb);

extern int index_values_strict_neg(const REAL *,INT nb);
extern int index_values_neg_or_null(const REAL *,INT nb);
extern int index_values_not_acos(const REAL *,INT nb);
extern int index_values_complex_nul(const REAL * x,const REAL *y,INT nb);

void  set_fonc_x(INT * res,INT x0,INT x1);

template <class Type>
        void compute_inside(INT * res,const Type * tx,const Type * ty,INT nb,
                            Type x0,Type y0,Type x1,Type y1);


template <class Type>
        void compute_inside(INT * res,const Type * tx,INT nb, Type x0,Type x1);

template <class Type>
        void compute_inside
             (
                   INT * res,
                   const Type * const *  coord,
                   INT nb,
                   INT dim,
                   const Type *p0,
                   const Type *p1
              );



template <class Type> Type red_tab_som(const Type * t,INT nb,Type v_init);


/*************************************************************/
/* SOME UTILS ON TAB                                         */
/*************************************************************/

template <class Type> inline Type ElAbs   (Type v1) {return ( (v1>0) ? v1 : -v1 );}
                      inline bool ElAbs   (bool v1) {return v1;}
inline REAL ecart_frac(REAL r)
{
    return ElAbs(signed_frac(r));
}
template <class Type> inline Type ElSquare(Type v1) {return (v1*v1);}

template <class Type> inline Type ElMax (Type v1,Type v2) {return (v1>v2) ? v1 : v2;}
template <class Type> inline Type ElMin (Type v1,Type v2) {return (v1<v2) ? v1 : v2;}

extern REAL VCube(REAL V);
extern REAL VPow4(REAL V);
extern REAL VPow5(REAL V);
extern REAL VPow6(REAL V);
extern REAL VPow7(REAL V);




template <class Type> inline Type ElMax3 (Type v1,Type v2,Type v3) {return ElMax(v1,ElMax(v2,v3));}
template <class Type> inline Type ElMin3 (Type v1,Type v2,Type v3) {return ElMin(v1,ElMin(v2,v3));}
template <class Type> inline Type ElMax4 (Type v1,Type v2,Type v3,Type v4)
                             {return ElMax(v1,ElMax3(v2,v3,v4));}
template <class Type> inline Type ElMin4 (Type v1,Type v2,Type v3,Type v4)
                             {return ElMin(v1,ElMin3(v2,v3,v4));}

template <class Type> inline void ElSwap (Type &v1,Type & v2)
        { Type  tmp = v1; v1 = v2; v2 = tmp; }
template <class Type> inline void set_min_max (Type &v1,Type & v2) {  if (v1 > v2) ElSwap(v1,v2);}


template <class Type,class T2> inline void ElSetMax (Type & v1,T2 v2) {if (v1<v2) v1=(Type)v2;}
template <class Type,class T2> inline void ElSetMin (Type & v1,T2 v2) {if (v1>v2) v1=(Type)v2;}

template <class Type> inline void SetInRange
                      (const Type & v0,Type & aV,const Type & v1)
{
     if (aV < v0)
        aV = v0;
     else if (aV> v1)
        aV = v1;
}

template <class Type> void elise_sort(Type *,INT);
template <class Type> void elise_indexe_sort(Type *,INT *,INT);


class cElRanGen
{
   public :
      REAL cNRrandom3 ();
      void cResetNRrand();
      REAL cNRrandC();
      cElRanGen();
      void InitOfTime(int aNb=1000);

   private :
      int inext,inextp;
      int MSEED;
      long ma[56];
      int iff;
      int idum ;
      float ran3 (int * idum);
};

class cRandNParmiQ
{
    public :
      cRandNParmiQ(int aN,int aQ);

      bool GetNext();
    private :
        int mN;
        int mQ;
};

std::vector<int> RandPermut(int aN);


extern void NRrandom3InitOfTime();
extern int  NRrandom3 (int aN);  // 0 <= X < N
extern REAL NRrandom3 ();
extern REAL NRrandC();  // entre -1 et 1
extern REAL NRrandInterv(double aV0,double aV1);  // entre -1 et 1
void ResetNRrand();
void rationnal_approx(REAL,INT &,INT&);


class ElTimer
{
     private :
        REAL _uval0;
        REAL _sval0;

        REAL _uval;
        REAL _sval;

        void  set_val();

     public :
        ElTimer();
        void reinit();
        REAL  uval();
        REAL  sval();
        REAL  ValAndInit();
        REAL  ValAbs();
};

REAL ElTimeOfDay();

class ElTabFlag
{
     public :
        ElTabFlag() : _flag(0) {}
        bool  kth(INT k) const     {return (_flag & (1<<k)) != 0 ;}
        void  set_kth_true(INT k)  {_flag |=  1<< k ; }
        void  set_kth_false(INT k) {_flag &=  (~ (1<< k)) ; }
        void  set_kth(INT k,bool val)
        {
              if (val)
                 set_kth_true(k);
              else
                 set_kth_false(k);
        }


     private :
        INT  _flag;
};

class ElFlagAllocator
{
      public :
         INT   flag_alloc();
         void  flag_free(INT);
      private :
          ElTabFlag _flag;
};

void CmpByEnd(const char * aName1,const char * aName2,INT & aK1,INT &aK2);
bool N2IsEndN1(const char * aName1,const char * aName2);


// Pour avoir un nom utilisable dans les commandes meme s'il
// contient des blancs
std::string ToStrBlkCorr(const std::string &);


std::string StrToLower(const std::string & aStr);
INT    IndPostfixed  (const ElSTDNS string &,char = '.');
ElSTDNS string StdPostfix(const ElSTDNS string &,char = '.');
ElSTDNS string StdPrefix (const ElSTDNS string &,char = '.');
ElSTDNS string StdPrefixGen (const ElSTDNS string &,char = '.');
std::string NameWithoutDir(const std::string &);

std::string ExtractDigit(const std::string & aName,const std::string &  aDef);


bool IsPrefix(const char * aPref,const char *aStr);


std::string ToCommande(int argc,char ** argv);
std::string QUOTE(const std::string & aStr);
void GlobStdAdapt2Crochet(std::string & aStr);

bool needPatternProtection( const string &aStr );
string PATTERN_QUOTE( const string &aStr );

bool SplitIn2ArroundCar
     (
         const std::string  &  a2Stplit,
         char                  aSpliCar,
         std::string  &  aBefore,
         std::string  &  aAfter,
         bool            AcceptNoCar  // Est on OK pour ne pas trouver aSpliCar
                                     // dans ce cas  aAfter est vide
     );

void  SplitInNArroundCar
      (
         const std::string  &  a2Stplit,
         char                  aSpliCar,
         std::string   &             aR0,
         std::vector<std::string>  &  aRAux
      );

void SplitIn2ArroundEq
     (
         const std::string  &  a2Stplit,
         std::string  &  aBefore,
         std::string  &  aAfter
     );

void SplitDirAndFile
     (
           std::string & aNameDir,
           std::string & aNameFile,
           const std::string & aStr
     );



std::vector<char *> ToArgMain(const std::string & aStr);



// Ajoute apres la dir et avant le .
std::string AddPrePost(const std::string & aName,const std::string & aPref,const std::string & aPost);

std::string DirOfFile(const std::string & aStr);

std::string StdWorkdDir(const std::string & aValWD,const std::string & aNameFile);

std::vector<std::string> VecStrFromFile(const std::string &);


bool GetOneModifLC
     (
         int argc,
         char ** argv,
         const std::string & aNameSymb,
         std::string &       aVal
     );

// RAJPOUTE DES /\ si necessaire
void MakeFileDirCompl(std::string &);


bool    IsPostfixed  (const ElSTDNS string &,char = '.');
bool    IsPostfixedBy  (const ElSTDNS string &,const std::string &);

bool IsFileDmp(const std::string &);


void EliseBRKP();

template <class Type> std::string ToString(const Type &); // util/string_dyn.cpp.o
std::string  ToStringNBD(int aNb,int aNbDig);



template <class Type>  std::istream & operator >> (std::istream &is,ElSTDNS vector<Type> & vec);

template <class Type> bool FromString(Type& x,const std::string & s)
{
   std::istringstream i(s);
   i >> x;
   return ! i.fail();
}

template <class Type> Type RequireFromString(const std::string & s,const std::string & aContext)
{
    Type aRes;
    bool Ok = FromString(aRes,s);
    if (! Ok)
    {
       std::cout << "Trying str=[" << s << "] in context :" << aContext << "\n";
       ELISE_ASSERT(false,"string is not a correc value for type");
    }

    return aRes;
}

template <class Type> int CmpTertiare(const Type & T1,const Type & T2)
{
   if (T1<T2) return -1;
   if (T1>T2) return  1;
   return 0;
}


class cElStatErreur
{
     public :
         cElStatErreur(INT NbValInit);
     void AddErreur(REAL);
     void Reset();
     REAL Erreur(REAL Pos) ; // Pos en 0.0 et 1.0,  Exemple :
                             // 0.0 = Vmin, 1.0=Vmax, 0.5 = Median etc..
     double  Avg() const;
     double  Ect() const;

     private :
        void    AssertNotEmpty() const;
        std::vector<REAL> mErrs;
        bool              mOk;
        REAL              mSom0;
        REAL              mSom1;
        REAL              mSom2;
};


template <class Type>
class cInterv1D
{
     public :
        cInterv1D(const Type & aV0,const Type & aV1);
        const Type & V0() const;
        const Type & V1() const;
        Type  Larg() const;
    cInterv1D<Type>  Inter(const cInterv1D<Type> &) const;
    cInterv1D<Type>  Dilate(const cInterv1D<Type> &) const;

     private:
        Type mV0;
        Type mV1;
    bool mEmpty;

};

class cDecoupageInterv1D
{
     public :
        cDecoupageInterv1D
        (
           const cInterv1D<int>  & aIntervGlob,
           int aSzMax,
           const cInterv1D<int>  & aSzBord,
           int                     anArrondi=1
        );
    int NbInterv() const;
    cInterv1D<int> KthIntervOut(int aK) const;
    // Avec Bord par defaut
    cInterv1D<int> KthIntervIn(int aK) const;
    cInterv1D<int> KthIntervIn(int aK,const cInterv1D<int>  & aSzBord) const;
    const cInterv1D<int> & IGlob() const;
    const cInterv1D<int> & IBrd() const;

    int LargMaxOut() const;
    int LargMaxIn(const cInterv1D<int>  & aSzBord) const;
    int LargMaxIn() const;


     private :
        int             KThBorneOut(int aK) const;

        cInterv1D<int>  mIntervGlob;
        cInterv1D<int>  mSzBord;
        int             mSzMax;
        int             mNbInterv;
        int             mArrondi;
};

template <class Type>
class cTplValGesInit
{
     public :
          cTplValGesInit() :
                // mVal(),  Bug Windows
                mIsInit(false)
          {
          }
          cTplValGesInit(const Type & aVal) :
                mVal(aVal),
                mIsInit(true)
          {
          }

/*
*/
      void SetNoInit() {mIsInit=false;}
          void SetVal(const Type & aVal) {mVal=aVal;mIsInit=true;}
          void SetValIfNotInit(const Type & aVal)
          {
               if (!mIsInit)
                  SetVal(aVal);
          }
          bool IsInit() const {return mIsInit;}
          Type & ValForcedForUnUmp() { return mVal; }
          void SetInitForUnUmp() {mIsInit=true;}
          const Type & ValForcedForUnUmp() const { return mVal; }
          const Type & Val() const
          {
              ELISE_ASSERT(mIsInit,"Unitialized Value in cValGesInit");
              return mVal;
          }
          const Type & Val(const std::string aMes ) const
          {
              if (!mIsInit)
              {
                  std::cout << "In context : " << aMes << "\n";
                  ELISE_ASSERT(false,"Unitialized Value in cValGesInit");
              }
              return mVal;
          }


          Type & Val()
          {
              ELISE_ASSERT(mIsInit,"Unitialized Value in cValGesInit");
              return mVal;
          }
          const Type & ValWithDef(const Type & aVal) const
          {
              return mIsInit ?mVal : aVal;
          }

          const Type * PtrVal() const { return mIsInit?&mVal:0;}
          const Type * PtrCopy() const { return mIsInit?new Type(mVal):0;}
          Type * PtrVal() { return mIsInit?&mVal:0;}
     private :
          Type mVal;
          bool mIsInit;
};


//typedef long long int tFileOffset;
typedef int64_t tLowLevelFileOffset;
typedef unsigned int  tByte4AbsFileOffset;
// typedef long long  int tLowLevelRelFileOffset;

class tFileOffset
{
    public :


         const tLowLevelFileOffset & CKK_AbsLLO() const
         {
               ELISE_ASSERT(mLLO.IsInit(),"AbsLLO :: NoInit");
               tLowLevelFileOffset aLLO = mLLO.Val();
               ELISE_ASSERT(aLLO>=0,"AbsLLO neg");
               return mLLO.Val();
         }
         tByte4AbsFileOffset   CKK_Byte4AbsLLO() const
         {
               ELISE_ASSERT(mLLO.IsInit(),"Byte4AbsLLO :: NoInit");
               tLowLevelFileOffset aLLO = mLLO.Val();
               ELISE_ASSERT((aLLO>=0) && (aLLO<=0xFFFFFFFFll),"Byt4LLO too big");
               return (tByte4AbsFileOffset)aLLO;
         }
         const tLowLevelFileOffset & BasicLLO() const
         {
               ELISE_ASSERT(mLLO.IsInit(),"BasicLLO :: NoInit");
               return mLLO.Val();
         }
         int  CKK_IntBasicLLO() const
         {
               ELISE_ASSERT(mLLO.IsInit(),"CKKBasicLLO :: NoInit");
               tLowLevelFileOffset aLLO = mLLO.Val();
               ELISE_ASSERT((aLLO>-0x7FFFFFFFll) && (aLLO<0x7FFFFFFFll),"Byt4LLO too big");
               return (int)aLLO;
         }

         tFileOffset ()
         {
             mLLO.SetNoInit();
         }
         tFileOffset (const tLowLevelFileOffset & aLLO) :
           mLLO(aLLO)
         {
         }

         tFileOffset operator + (const tFileOffset & anO2) const
         {
               return mLLO.Val() + anO2.mLLO.Val();
         }
         tFileOffset operator - (const tFileOffset & anO2) const
         {
               return mLLO.Val() - anO2.mLLO.Val();
         }
         tFileOffset operator / (const tFileOffset & anO2) const
         {
               return mLLO.Val() / anO2.mLLO.Val();
         }
         tFileOffset operator * (const tFileOffset & anO2) const
         {
               return mLLO.Val() * anO2.mLLO.Val();
         }

         bool operator < (const tFileOffset & anO2) const
         {
               return mLLO.Val() < anO2.mLLO.Val();
         }
         bool operator > (const tFileOffset & anO2) const
         {
               return mLLO.Val() > anO2.mLLO.Val();
         }
         bool operator == (const tFileOffset & anO2) const
         {
               return mLLO.Val() == anO2.mLLO.Val();
         }
         bool operator != (const tFileOffset & anO2) const
         {
               return mLLO.Val() != anO2.mLLO.Val();
         }

         void operator ++ (int)
         {
              mLLO.SetVal(mLLO.Val()+1);
         }
         void operator +=  (const tFileOffset & anO2)
         {
              mLLO.SetVal(mLLO.Val()+anO2.mLLO.Val());
         }
         void operator -=  (const tFileOffset & anO2)
         {
              mLLO.SetVal(mLLO.Val()-anO2.mLLO.Val());
         }
         void operator *=  (const tFileOffset & anO2)
         {
              mLLO.SetVal(mLLO.Val()*anO2.mLLO.Val());
         }


/*
         void SetLLO(const tLowLevelFileOffset & aLLO)
         {
              mLLO.SetVal(aLLO);
         }
*/
         bool IsInit() const
         {
              return mLLO.IsInit();
         }

// Deux interface bas niveaus, "tres sales", poiur assurer la communication avec le stockage
// en int des offset dans les tiffs qui est necessaire pour utiliser le service de tag generiques
         static  tFileOffset CKK_FromReinterpretInt(int anI)
         {
               tByte4AbsFileOffset anUI;
               memcpy(&anUI,&anI,sizeof(tByte4AbsFileOffset));
               return tFileOffset(anUI);
         }
         int CKK_ToReinterpretInt() const
         {
              int aRes;
              tByte4AbsFileOffset anOfs4 = CKK_Byte4AbsLLO();
              memcpy(&aRes,&anOfs4,sizeof(tByte4AbsFileOffset));
              return aRes;
         }

         static  const tFileOffset NoOffset;
/*
*/
    private :
        cTplValGesInit<tLowLevelFileOffset> mLLO;
};

inline std::ostream & operator << (std::ostream & ofs,const tFileOffset  &anOffs)
{
    ofs << anOffs.BasicLLO();
    return ofs;
}

typedef tFileOffset tRelFileOffset;


// typedef unsigned int tFileOffset;
/*
*/


tFileOffset RelToAbs(tRelFileOffset anOff);


/*****************************************************/
/*                                                   */
/*                  cEquiv1D                      */
/*                                                   */
/*****************************************************/

// Classe pour gerer rapidement les "classe equiv 1D"
// c'est a dire les intervalles classes d'equivalences
// d'une application croissante  de Z dans Z

class cFonc1D
{
    public :
         virtual int operator()(const int &) const = 0;
    virtual ~cFonc1D() {}
    private :
};

class cFonc1D_HomTr : public cFonc1D
{
    public :
         //   (a X +b ) / c
         //  vraie division   (Elise_div)
         //  requiert : c > 0
         int operator()(const int &) const;
         cFonc1D_HomTr(const int & anA,const int & aB,const int & aC);
    private :
         int mA;
         int mB;
         int mC;
};


template<class aType> class cVectTr;

class cEquiv1D
{
     public :
          // Intervalle [aV0 aV1[
          cEquiv1D ();
          void InitFromFctr
          (
                 int aV0,
                 int aV1,
                 const cFonc1D &
          );
          void InitByFusion(const cEquiv1D &,int aFus);
          // InitByFusion, vs constructeur
          class cCstrFusion {};
          cEquiv1D(const cCstrFusion &,const cEquiv1D &,int aFus);

          void InitByClipAndTr
               (
                    const cEquiv1D &,
                    int aHomOfNewV0,
                    int aNewV0,
                    int aNewV1
               );
          void InitByDeZoom
               (
                    const cEquiv1D &,
                    int aDz,
                    cVectTr<int> * mLut
               );
         int NbClasses() const { return mNbClasses; }
          int V0()        const { return mV0; }
          int V1()        const { return mV1; }
          int  NumClasse(const int & aV) const
          {
               return mNumOfClasse[aV-mV0];
          }
          // La fonction "inverse" renvoie un intervalle
          void ClasseOfNum(int & aV0,int & aV1,const int & aNCL) const
          {
                aV0 = mDebOfClasse[aNCL];
                aV1 = mDebOfClasse[aNCL+1];
          }
          int SzMaxClasses() const;
     private:
          void Reset(int aV0,int aV1);
          std::vector<int> mNumOfClasse; // Pour un entier, son numero de classe
          std::vector<int> mDebOfClasse; // Pour un numero de classe, sont entier de debut

          int mV0;
          int mV1;
          int mNbClasses;
};

// GPAO : Gestion de Production assistee par ordinateur. C'est
// un peu presenptueux pour l'instant il s'agit juste d'avoir des
// classes permettant de gerer du calcul distribue avec des regles
// gerable par des DAG (directed acyclique graphe)
//
//  cEl_GPAO  est la classe "manager"
//

class cEl_GPAO;
class cEl_Task;

//#include "cElCommand.h"

class cElTask
{
     public :
          void AddDep(cElTask &);
          void AddDep(const std::string &);  // Idem AddDep(cElTask &)
          void AddBR(const std::string &);  //
          void  GenerateMakeFile(FILE *) const;
          // Genere le mkf, l'execute, le purge
     private :
          friend class cEl_GPAO;
        #ifdef __USE_EL_COMMAND__
          cElTask
          (
               const std::string & aName,
               cEl_GPAO &,
               const cElCommand & aBuildingRule
          );
        #else
          cElTask
          (
               const std::string & aName,
               cEl_GPAO &,
               const std::string & aBuildingRule
          );
        #endif
         cEl_GPAO &  mGPAO;

         std::string mName;
        #ifdef __USE_EL_COMMAND__
            std::list<cElCommand> mBR;  // BuildingRule
        #else
            std::list<std::string> mBR;  // BuildingRule
        #endif

         std::vector<cElTask *>  mDeps;
};


class cEl_GPAO
{
     public :
          // Interface simplifiee quand il n'y a pas de dependance entre les commandes
          static void DoComInParal(const std::list<std::string> &,std::string  FileMk = "", int   aNbProc = 0 ,bool Exe=true, bool MoinsK=false);
          static void DoComInSerie(const std::list<std::string> &);

         ~cEl_GPAO();
          cEl_GPAO();

        #ifdef __USE_EL_COMMAND__
            cElTask   & NewTask
                        (
                            const std::string &aName,
                            const cElCommand & aBuildingRule
                        ) ;

            cElTask   & GetOrCreate
                        (
                            const std::string &aName,
                            const cElCommand & aBuildingRule
                        );
        #else
            cElTask   & NewTask
                        (
                                const std::string &aName,
                                const std::string & aBuildingRule
                            ) ;

            cElTask   & GetOrCreate
                        (
                                const std::string &aName,
                                const std::string & aBuildingRule
                            ) ;
        #endif


         cElTask   &TaskOfName(const std::string &aName) ;
         void  GenerateMakeFile(const std::string & aNameFile) const ;
         void  GenerateMakeFile(const std::string & aNameFile,bool ModeAdditif) const;
         void ExeParal(std::string aFile,int aNbProc = -1,bool SuprFile=true);
         void dump( std::ostream &io_ostream=std::cout ) const;
     private :
         std::map<std::string,cElTask *>  mDico;

};


//  Pour executer une commande en // sur +sieur fichier, pour l'instant on fait
// basique, on ajoutera eventuellement apres des cles avec cInterfChantierNameManipulateur
/*
void MkFMapCmd
     (
          const std::string & aBefore,
          const std::vector<std::string> aSet ,
          const std::string & anAfter
     );
*/

void MkFMapCmdFileCoul8B
     (
          const std::string & aDir,
          const std::vector<std::string > &aSet
     );



//========================================================

class cInterfChantierNameManipulateur;

void RequireBin
     (
         const std::string & ThisBin,  // Le prog appelant pour evt
         const std::string & BinRequired,
     const std::string & LeMake = "Makefile"  // Si
     );

// For top call like Tapas, Malt , .. want to duplicate args in @
int TopSystem(const std::string & aComOri);

#define DEF_SVP_System false
#define DEF_AdaptGlob_System false

int System(const std::string & aCom,bool aSVP=DEF_SVP_System,bool AddOptGlob=DEF_AdaptGlob_System,bool UseTheNbIterProcess=false);

void  EliseVerifAndParseArgcArgv(int argc,char ** argv);


class cAppliBatch
{
    public :
       typedef enum
       {
            eExeDoNothing,
            eExeDoIfFileDontExist,
            eExeDoSys,
            eExeWriteBatch
       } eModeExecution;

       typedef enum
       {
             eNoPurge =0,
             ePurgeTmp =1,
             ePurgeAll =2
       } eNivPurge;


       void DoAll();

       //const std::string & ThisBin() const;

       cEl_GPAO &  GPAO ();
       bool        ByMKf() const;  // By Make file
       const std::string & MKf() const;
       // Parfois le plus simple est que le programme se rappelle lui - meme avec
       // des option legerement differente dans ce cas on doit etre au courant
       bool        IsRelancedByThis() const;

    protected :
        virtual ~cAppliBatch();
        cAppliBatch
    (
         int,
         char **,
         int aNbArgGlob,
         int aNbFile,
         const std::string & aPostFixWorkDir,
         const std::string & aKeyDOIDE="",
             bool  ForceByDico = false
        );

     void AddPatSauv(const std::string &);
         int ARGC();
     char ** ARGV();
     std::string ComCommune() const;
         int System(const std::string &,bool aSVP=false);
         int System(const char* FileCible,const std::string &,bool aSVP=false);
     const std::string & CurF1() const;
     const std::string & CurF2() const;
     const std::string & CurF(int aK) const;
         const std::string  & DirChantier() const;
         const std::string  & DirTmp() const;
         const std::string  & DirSauv() const;
     cInterfChantierNameManipulateur * ICNM();
     cInterfChantierNameManipulateur * ICNM() const;

     bool NivPurgeIsInit();
     void SetNivPurge(eNivPurge  );
     bool NivExeIsInit();
     void SetNivExe(eModeExecution);
         eModeExecution ModeExe() const;
         std::string ComForRelance();
    std::string protectFilename( const std::string &i_filename ) const; // according to ByMKf()

    private :
    void DoOne();
    virtual void Exec() = 0;
    void UseLFile(const std::list<std::string> &);



    // Les args non consommes

     // Partie de la ligne de commande qui revient a chaque fois


    // private :
    std::vector<char *>  mArgsNC;  //

        void DoPurge();

    cInterfChantierNameManipulateur * mICNM;

        //std::string  mThisBin;
        std::string  mDirChantier;

    std::string  mPostFixWorkDir;
    int          mNbFile;
        bool         mByNameFile;
    std::string  mDirSauv;
    std::string  mDirTmp;

        //bool         mFileByICNM;
    std::string  mPatF1;
    std::string  mPatF2;
    std::string  mCurF1;
    std::string  mCurF2;
    std::vector<std::string> mVCurF;

    std::string  mArgAdd;

        eModeExecution  mModeExe;
    bool            mExeIsInit;
    eNivPurge       mNivPurge;
    bool            mNivPurgeIsInit;
        std::string     mFileBatch;
    std::vector<std::string>  mPatSauv;

    bool                      mFileByPat;
    bool                      mByDico;
    std::list<std::string>    mListFile1ByPat;
    int  mReverse;
    int                       mDOIDE;
    std::string               mKeyDOIDE;

        std::string               mMKf;
        bool                      mModeAddMkf;
        int                       mIsRelancedByThis;
        std::string               mDebCom;
        std::string               mEndCom;
        cEl_GPAO                  mGPAO;
};

class cCpleString
{
     public :
        cCpleString AddPrePost
                    (
                        const std::string& aPre,
                        const std::string& aPost
                    ) const;
        cCpleString(const std::string&,const std::string&);
        cCpleString();
    const std::string &  N1() const;
    const std::string &  N2() const;

    bool operator < (const cCpleString &) const;
    bool operator == (const cCpleString &) const;
     private :
        std::string mN1;
        std::string mN2;
};

class cMonomXY
{
     public :
         cMonomXY(double,int,int);
         cMonomXY();
         double mCoeff; 
         int mDegX; 
         int mDegY; 
};

class cXmlHour;
class cXmlDate;

class cElHour
{
    public :
      cXmlHour ToXml();
      static cElHour FromXml(const cXmlHour &);
      cElHour
      (
          int aNbHour,
          int aNbMin,
      double aNbSec
      );
      double InSec() const; // Sec depuis minuits
      int    H() const;
      int    M() const;
      double S() const;

      bool operator==( const cElHour &i_b ) const;
      bool operator!=( const cElHour &i_b ) const;

     // read/write in raw binary format
     void from_raw_data( const char *&io_rawData, bool i_reverseByteOrder );
     void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
     static unsigned int raw_size();

     void read_raw( istream &io_istream, bool i_inverseByteOrder );
     void write_raw( ostream &io_ostream, bool i_inverseByteOrder ) const;

     static void getCurrentHour_local( cElHour &o_localHour );
     static void getCurrentHour_UTC( cElHour &o_utcHour );

    private :
       int mH;
       int mM;
       double mS;
};

ostream & operator <<( ostream &aStream, const cElHour &aHour );

class cElDate
{
    public :
       cXmlDate ToXml();
       static cElDate FromXml(const cXmlDate &);

       static const cElDate NoDate;
       bool IsNoDate() const;
       cElDate
       (
         int aDay,
         int aMonth,
         int aYear,
     const cElHour &
       );

        const cElHour &  H() const;
    int Y() const;
    int M() const;
    int D() const;

    static cElDate FromString(const std::string &);

    int NbDayFrom1erJ() const;
    // Ne prend pas en compte les 13 jours "sautes " au 17e
    int NbDayFromJC() const;
    int    DifInDay(const cElDate&) const;
    double DifInSec(const cElDate&) const;

    bool operator==( const cElDate &i_b ) const;
    bool operator!=( const cElDate &i_b ) const;

    // read/write in raw binary format
    void from_raw_data( const char *&io_rawData, bool i_reverseByteOrder );
    void to_raw_data( bool i_reverseByteOrder, char *&o_rawData ) const;
    static unsigned int raw_size();

    void read_raw( istream &io_istream, bool i_inverseByteOrder=false );
    void write_raw( ostream &io_ostream, bool i_inverseByteOrder=false ) const;

    static void getCurrentDate_local( cElDate &o_localDate );
    static void getCurrentDate_UTC( cElDate &o_utcDate );

    private :
         int mD;
         int mM;
         int mY;
     cElHour mH;

         // !! Les mois commencent a 1
     static const int TheNonBisLengthMonth[12];
     static int TheNonBisLengthMonthCum[12];
     static int TheBisLengthMonthCum[12];
     static bool TheIsBis[3000];
     static int  TheNbDayFromJC[3000];

     static bool mTabuliIsInit;

     static void InitTabul();


     static bool PrivIsBissextile(int aY);
};

bool operator < (const cElDate & aD1, const cElDate & aD2);

ostream & operator <<( ostream &aStream, const cElDate &aDate );

class cINT8ImplemSetInt
{
    public :
       cINT8ImplemSetInt();
       void Add(int anInt);
       bool IsIn (int anInt) const;
       int  NumOrdre(int aI) const;  // Nombre d'Entier < a aI
       int  NumOrdre(int aI,bool svp) const;  // Nombre d'Entier < a aI
       static int  Capacite();

       bool operator < (const cINT8ImplemSetInt &) const;
    private :
       INTByte8  mFlag;
};

template <const int NbI> class cSetIntMultiple
{
    public :
       cSetIntMultiple();
       void Add(int anInt);
       bool IsIn (int anInt) const;
       int  NumOrdre(int aI) const;  // Nombre d'Entier < a aI
       static int  Capacite();

       bool operator < (const cSetIntMultiple<NbI> &) const;
    private :
        Pt2d<INT>  NumSub(const int & anI) const; // x le set, y le I dans le set
       cINT8ImplemSetInt  mSets[NbI];
};

class cVarSetIntMultiple
{
    public :
       cVarSetIntMultiple();
       void Add(int anInt);
       bool IsIn (int anInt) const;
       int  NumOrdre(int aI) const;  // Nombre d'Entier < a aI

       bool operator < (const cVarSetIntMultiple &) const;
       int  Capacite() const;
    private :
        Pt2d<INT>  NumSub(const int & anI) const; // x le set, y le I dans le set
        mutable std::vector<cINT8ImplemSetInt>  mSets;
};



class cElXMLTree;

extern bool TransFormArgKey
     (
         std::string & aName ,
         bool AMMNoArg,  // Accept mismatch si DirExt vide
         const std::vector<std::string> & aDirExt
     );

// Class sepeciale pour gerer les objets autre que string qui peuvent etre initialise par des #1 #2 ..
// dans les xml pour les cles parametrees, par exemple le <DeltaMin> de <ByAdjacence>

template <class Type> class TypeSubst
{
    public :
          TypeSubst();
          TypeSubst(const Type& Val);
          void SetStr(cElXMLTree *);
          const  Type  & Val() const;
          bool  Subst(bool AMMNoArg,  const std::vector<std::string> & aVParam);
          void TenteInit();

    private :
          Type           mVal;
          bool           mIsInit;
          std::string    mStrInit;
          std::string    mStrTag;
};

typedef TypeSubst<bool>    BoolSubst;
typedef TypeSubst<int>     IntSubst;
typedef TypeSubst<double>  DoubleSubst;

#if __cplusplus <= 199711L
template <class T> T* VData(std::vector<T> & aV)  {return &(aV[0]);}
template <class T> const T* VData(const std::vector<T> & aV)  {return &(aV[0]);}
#else
template <class T> T* VData(std::vector<T> & aV)  {return aV.data();}
template <class T> const T* VData(const std::vector<T> & aV)  {return aV.data();}
#endif

///  Ajoute des regles speciales pour que chaque pixle ait au moins un 
//  precedcesseur et un antecedant
//   Z est dans l'intervalle ouvert I1 [aZ1Min,aZ1Max[,

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
              );

///  Ne force pas les connexions
void BasicComputeIntervaleDelta
              (
                  INT & aDzMin,
                  INT & aDzMax,
                  INT aZ,
                  INT MaxDeltaZ,
                  INT aZ0Min,
                  INT aZ0Max
              );
double FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec);

std::string getBanniereMM3D();

void BanniereMM3D();



extern "C" {
FILE * ElFopen(const char *path, const char *mode);
int ElFclose(FILE *fp);
void ShowFClose();

};


void GetSubset(std::vector<std::vector<int> > & aRes,int aNb,int aMax);

bool ElGetStrSys( const std::string & i_base_cmd, std::string &o_result );

void BanniereGlobale();

// protect spaces with backslashes (for use with 'make')
string protect_spaces( const string &i_str );

int MMNbProc();

// lanch the "make" program
// do not include "-j x" flag in i_options, it is handle by i_nbJobs
// i_nbJobs = 0 means "-j" (i.e. infinite jobs, which is not recommanded)
// i_rule can be an empty string, if so, make will launch the makefile's default rule
// returns make's return code
bool launchMake( const std::string &i_makefile, const std::string &i_rule=std::string(), unsigned int i_nbJobs=MMNbProc(), const std::string &i_options=std::string(), bool i_stopCurrentProgramOnFail=true );


double MoyHarmonik(const double & aV1,const double & aV2);
double MoyHarmonik(const double & aV1,const double & aV2,const double & aV3);

size_t getSystemMemory();

size_t getUsedMemory();

std::string humanReadable( size_t aSize );

template <class T>
inline char toS(const T &v)
{
	return (v < 2 ? '\0' : 's');
}

// GIT Revision : {last tag}-{nb commit since the last tag}-{id of the last commit}-{dirty if modified since the last commit}
// ex: version_1.0.beta4-1-gbd6bc8d-dirty
std::string gitRevision();

#endif /* ! _ELISE_UTIL_H */



/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
