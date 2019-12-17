#include "include/MMVII_all.h"
#include "ExternalInclude/Eigen/Dense"


namespace MMVII
{
/** \file TestMyJets.cpp
    \brief Make some benchmark on jets (ceres) 

    This file contain test on efficiency of jets. First draft with 
  home made implemntation

*/

/*
   Fab(X0,X1) = 1/(aX0 + b X1^2)

   Fab(X0+e0,X1+e1) = 1/(a(X0+e1) + b(X1+e1)^2)
                    = 1 / (aX0 + ae1 + b(X1^2 + 2Xe1))
                    = 1 / (aX0+b
*/

/* ************************************** */
/*                                        */
/*           cVarEpsNum<int N>            */
/*                                        */
/* ************************************** */

struct cVarEpsNum 
{
     public  :
        cVarEpsNum(double aNum) :
            mNum  (aNum)
        {
           Init();
        }

        void AddVal(double aVal,int aK) 
        {
             mVEps.push_back(aVal);
             mVInd.push_back(aK);
        }
        cVarEpsNum(double aNum,int aK) :
            mNum  (aNum)
        {
              AddVal(1.0,aK);
             // mVEps.push_back(1.0);
             // mVInd.push_back(aK);
        }

        cVarEpsNum(const cVarEpsNum & aVEN,double aNum,double aMul) :
            mNum  (aNum),
            mVEps (aVEN.mVEps),
            mVInd (aVEN.mVInd)
        {
             for (auto & aEps : mVEps)
                 aEps *= aMul;
        }


        double               mNum;
        std::vector<double>  mVEps;
        std::vector<int>     mVInd;
        cVarEpsNum operator+(const cVarEpsNum& g)  const;
        cVarEpsNum operator*(const cVarEpsNum& g)  const;


        static const int SzBuf = 1000;
        static int    BufOccupied[SzBuf];
        static double BufEps[SzBuf];
        static bool   IsInit;

        static void Init()
        {
            if (IsInit) return;
            IsInit = true;
            for (int aK=0 ; aK<SzBuf ; aK++)
            {
                BufOccupied[aK] = false;
                BufEps[aK] = false;
            }
        }
        inline void AddToBuf() const;
        inline void AddToBuf(const double & aMul) const;
        inline void BuffAdd(cVarEpsNum &) const;
};

int    cVarEpsNum::BufOccupied[SzBuf];
double cVarEpsNum::BufEps[SzBuf];
bool   cVarEpsNum::IsInit = false;

// ============= operator + ======================

void cVarEpsNum::AddToBuf() const
{
   for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
   {
       BufEps[mVInd[aK] ]    += mVEps[aK];
   }
}
void cVarEpsNum::AddToBuf(const double & aMul) const
{
   for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
   {
       BufEps[mVInd[aK]] += mVEps[aK] * aMul;
   }
}

void cVarEpsNum::BuffAdd(cVarEpsNum & aRes) const
{
   for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
   {
       int IndF = mVInd[aK];
       double & aVal = BufEps[IndF];
       if ( aVal)
       {
           aRes.AddVal(aVal,IndF);
           aVal = 0;
       }
   }
}
   // =========== cVarEpsNum  operator unaire ==============

inline cVarEpsNum COS(const cVarEpsNum& g)  
{
    return cVarEpsNum(g,cos(g.mNum),-sin(g.mNum));
}
   // =========== cVarEpsNum  operator * ==============

inline cVarEpsNum cVarEpsNum::operator*(const cVarEpsNum& g)  const
{
   cVarEpsNum aRes(mNum*g.mNum);

   AddToBuf(g.mNum);
   g.AddToBuf(mNum);

   BuffAdd(aRes);
   g.BuffAdd(aRes);

   return aRes;
}
inline cVarEpsNum operator*(const cVarEpsNum& f,const  double &  aV) 
{
    cVarEpsNum aRes(f);
    aRes.mNum *= aV;
    for (auto & aEps : aRes.mVEps)
       aEps *= aV;
    return aRes;
}
inline cVarEpsNum operator*(const double & aV,const cVarEpsNum& f) {return f*aV;}


   // =========== cVarEpsNum  operator + ==============

inline cVarEpsNum operator+(const cVarEpsNum& f,const  double &  aV) 
{
    cVarEpsNum aRes(f);
    aRes.mNum += aV;
    return aRes;
}
inline cVarEpsNum operator+(const double & aV,const cVarEpsNum& f) {return f+aV;}

inline cVarEpsNum cVarEpsNum::operator+(const cVarEpsNum& g)  const
{
   cVarEpsNum aRes(mNum+g.mNum);

   AddToBuf();
   g.AddToBuf();

   BuffAdd(aRes);
   g.BuffAdd(aRes);
   
/*
   for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
   {
       int Indf = mVInd[aK];
       BufOccupied[Indf ] = true;
       BufEps[Indf ]      = mVEps[aK];
   }

   aRes = g;
   for (unsigned int aK=0 ; aK<g.mVEps.size(); aK++)
   {
       int Indg = g.mVInd[aK];
       if (BufOccupied[Indg])
       {
           BufOccupied[Indg] = false;
           aRes.mVEps[aK] += BufEps[Indg];
       }
   }

   for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
   {
       int Indf = mVInd[aK];
       if (BufOccupied[Indf ])
       {
          BufOccupied[Indf] = false;
          aRes.mVEps.push_back(BufEps[aK]);
          aRes.mVInd.push_back(Indf);
       }
   }
*/

   return aRes;
}

/* ************************************** */
/*                                        */
/*           cEpsNum<int N>               */
/*                                        */
/* ************************************** */

template<int N> struct cEpsNum {
  double mNum;
  Eigen::Matrix<double, 1, N> mEps;
  cEpsNum(const double  & aNum,const Eigen::Matrix<double, 1, N> & aEps) :
     mNum (aNum),
     mEps (aEps)
  {
  }
  cEpsNum(const double  & aNum,int aK) :
      mNum  (aNum),
      mEps  (Eigen::Matrix<double, 1, N>::Zero())
  {
      mEps(aK) = 1.0;
  }
  cEpsNum(double aNum) :
      mNum  (aNum),
      mEps  (Eigen::Matrix<double, 1, N>::Zero())
  {
  }
  cEpsNum() :
     cEpsNum(0.0)
  {
  }
  cVarEpsNum  ToVEN() const;
  static cEpsNum<N>  Random(double Densite);
};

template<int N> cEpsNum<N>   cEpsNum<N>::Random(double Densite)
{
    cEpsNum<N> aRes(N*RandUnif_C());
    for (int aK=0 ; aK<N ; aK++)
    {
       if (RandUnif_0_1() < Densite)
       {
          aRes.mEps[aK] = RandUnif_C();
       }
    }
    return aRes;
}

template<int N> cVarEpsNum   cEpsNum<N>::ToVEN() const
{
    cVarEpsNum aRes(mNum);
    for (int aK=0 ; aK<N ; aK++)
        if (mEps[aK])
           aRes.AddVal(mEps[aK],aK);
    return aRes;
}
/*
*/

template<int N> double EpsDifference(const cEpsNum<N> & aEps,const cVarEpsNum & aVarEps)
{
    cVarEpsNum::Init();
    // double aRes=0.0;
    double aRes= std::abs(aEps.mNum - aVarEps.mNum);
    cEpsNum<N>  aEps2;
    
    for (unsigned int aK=0 ; aK<aVarEps.mVEps.size() ; aK++)
    {
         int  aInd = aVarEps.mVInd[aK];
         double aVal = aVarEps.mVEps[aK];
         if (aInd>=N) 
            aRes += std::abs(aVal);
         else
            aEps2.mEps[aInd] += aVal;
    }
    for (int aK=0 ; aK<N ; aK++)
       aRes += std::abs(aEps2.mEps[aK]-aEps.mEps[aK]);

    return aRes;
}


typedef double (*OpDouble) (double aVal);

template<int N> Eigen::Matrix<double, 1, N> ApplyOp(const Eigen::Matrix<double, 1, N> aMat,OpDouble anOp)
{
   Eigen::Matrix<double, 1, N> aRes;
   for (int aK=0 ; aK<N; aK++)
       aRes(aK)  = anOp(aMat(aK));
   return aRes;
}

     // ====== operator + =============

template<int N> cEpsNum<N> operator+(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum + g.mNum, f.mEps + g.mEps);
}

template<int N> cEpsNum<N> operator+(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f + g.mNum, g.mEps);
}
template<int N> cEpsNum<N> operator+(const cEpsNum<N>& g,const double & f) {
  return f+g;
}
     // ====== operator - =============

template<int N> cEpsNum<N> operator-(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum - g.mNum, f.mEps - g.mEps);
}

template<int N> cEpsNum<N> operator-(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f - g.mNum, -g.mEps);
}
template<int N> cEpsNum<N> operator-(const cEpsNum<N>& g,const double & f) {
  return cEpsNum<N>(g.mNum -f, g.mEps);
}

     // ====== operator * =============

template<int N> cEpsNum<N> operator*(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum * g.mNum,  g.mNum * f.mEps + f.mNum * g.mEps);
}
template<int N> cEpsNum<N> operator*(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f*g.mNum,f*g.mEps);
}
template<int N> cEpsNum<N> operator*(const cEpsNum<N>& g,const double & f) {
  return f*g;
}




template<int N> cEpsNum<N> operator/(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum / g.mNum,  (f.mEps / g.mNum) - g.mEps *(f.mNum/Square(g.mNum)));
}

template<int N> cEpsNum<N> Square(const cEpsNum<N>& f) {
  return cEpsNum<N>(Square(f.mNum) , 2* f.mNum* f.mEps  );
}

template<int N> cEpsNum<N> Cube(const cEpsNum<N>& f) {
  return cEpsNum<N>(Cube(f.mNum) , (3*Square(f.mNum)) * f.mEps );
}




template<int N> cEpsNum<N> COS(const cEpsNum<N>& f) {
  return cEpsNum<N>(cos(f.mNum),-sin(f.mNum)*f.mEps);
}
inline double COS(const double & v) {return cos(v);}

template <class Type>  class cProjCamRad 
{
    public :
       void Compute (Type * Parameter,Type * Residual);

        cProjCamRad();
        cDenseMatrix<double>  mRotCur;
        cPt2dr  mPix;

};
class  cJetsTestCam : public cInterfaceTestCam
{
    public : 
       void  InitFromParams(const std::vector<double> &) override;
       void  Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & ) override;
       void  Compute(int aNb) override;
    private : 
       cProjCamRad<cEpsNum<16> > mJetsCam;

};

std::vector<double>    StdParamTestCam(double AmplNoise)
{
    std::vector<double> aRes;

    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //X-Gr
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Y-Gr
    aRes.push_back(1.0+0.01*RandUnif_C()*AmplNoise);  //Z-Gr

    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //X-Cam
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Y-Cam
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Z-Cam

    aRes.push_back(0.0);   // W-x   Mandotary 000 as it is the complementary rotation
    aRes.push_back(0.0);   // W-y
    aRes.push_back(0.0);   // W-z

    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // Centre dist X
    aRes.push_back(0.02 *RandUnif_C()*AmplNoise);   // Centre dist Y


    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K1
    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K2
    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K3


    aRes.push_back(3000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPx
    aRes.push_back(2000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPy
    aRes.push_back(5000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPz / Focale

    return aRes;

}


template <class Type>  cProjCamRad<Type>::cProjCamRad() :
    mRotCur(3,3,eModeInitImage::eMIA_MatrixId)
{
}
// std::vector<Type> InitParam();

template <class Type>  void cProjCamRad<Type>::Compute(Type * Parameter,Type * Residu)
{
    // Ground Coordinates of projected point
    Type & XTer = Parameter[0];
    Type & YTer = Parameter[1];
    Type & ZTer = Parameter[2];

    // Coordinate of camera center
    Type & C_XCam = Parameter[3];
    Type & C_YCam = Parameter[4];
    Type & C_ZCam = Parameter[5];

    // Coordinate of Omega vector coding the unknown "tiny" rotation
    Type & Wx = Parameter[6];
    Type & Wy = Parameter[7];
    Type & Wz = Parameter[8];

    // Coordinate Center of distorstion
    Type & xCD = Parameter[9];
    Type & yCD = Parameter[10];

    // Distortions coefficients
    Type & k2D = Parameter[11];
    Type & k4D = Parameter[12];
    Type & k6D = Parameter[13];

    // PP and Focal
    Type & xPP = Parameter[14];
    Type & yPP = Parameter[15];
    Type & zPP = Parameter[16]; // also named as focal

    // Vector P->Cam
    Type  XPC = XTer-C_XCam;
    Type  YPC = YTer-C_YCam;
    Type  ZPC = ZTer-C_ZCam;

    // Coordinate of points in  camera coordinate system, do not integrate "tiny" rotation

    Type  XCam0 = mRotCur(0,0)*XPC +  mRotCur(1,0)*YPC +  mRotCur(2,0)*ZPC;
    Type  YCam0 = mRotCur(0,1)*XPC +  mRotCur(1,1)*YPC +  mRotCur(2,1)*ZPC;
    Type  ZCam0 = mRotCur(0,2)*XPC +  mRotCur(1,2)*YPC +  mRotCur(2,2)*ZPC;

     
    //  Wx      X      Wy * Z - Wz * Y
    //  Wy  ^   Y  =   Wz * X - Wx * Z
    //  Wz      Z      Wx * Y - Wy * X
   
     //  P =  P0 + W ^ P0 
    
    Type  XCam = XCam0 + Wy * ZCam0 - Wz * YCam0;
    Type  YCam = YCam0 + Wz * XCam0 - Wx * ZCam0;
    Type  ZCam = ZCam0 + Wx * YCam0 - Wy * XCam0;

    // Projection 

    Type xPi =  XCam/ZCam;
    Type yPi =  YCam/ZCam;


    // Coordinate relative to distorsion center
    Type xC =  xPi-xCD;
    Type yC =  yPi-yCD;
    Type Rho2C = Square(xC) + Square(yC);

   // Compute the distorsion
    Type Dist = k2D*Rho2C + k4D * Square(Rho2C) + k6D*Cube(Rho2C);
    
    Type xDist =  xPi + xC * Dist;
    Type yDist =  xPi + yC * Dist;
    
   // Use principal point and focal
    Type xIm =  xPP  + zPP  * xDist;
    Type yIm =  yPP  + zPP  * yDist;

    Residu[0] = xIm - mPix.x();
    Residu[1] = yIm - mPix.y();
}

// class cProjCamRad<double>;
// class cProjCamRad<cEpsNum<6> >;


template <class Type,int Nb>  class cTestJets
{
    public :
       Type   ComputeExpansed(Type * Parameter) const;
       Type   ComputeLoop(Type * Parameter) const;
       double  Phase[Nb];
       double  Freq[Nb];

       cTestJets() ;
};

///  TJ(K) (COS(Freq[K]*Square(Param[K]+0.2) + Phase[K]))

/*
template <int Nb>  cEpsNum<Nb> TJ_DerAnalytique(const cTestJets<double,Nb> & aTJ,double * aParam)
{
    const double* Phase = aTJ.Phase;
    const double* Freq  = aTJ.Freq;
    double Val = aTJ.ComputeExpansed(aParam);
    double V2 = Square(Val);
    Eigen::Matrix<double, 1, Nb> mEps;
    for (int aK=0 ; aK<Nb ; aK++)
    {
       double aP2= aParam[aK]+0.2;
       mEps[aK] = 2*aP2*Freq[aK] *sin(Freq[aK]*Square(aP2)+Phase[aK]) *V2;
    }

    return cEpsNum<Nb>(Val,mEps);
}
*/

///  TJ(K) (COS(Freq[K]*Square(Param[K]+0.2) + Phase[K]))
template <int Nb>  cEpsNum<Nb> TJ_DerAnalytique(const cTestJets<double,Nb> & aTJ,double * aParam)
{
    static double Tmp[Nb];
    double Som = 0.0;
    const double* Phase = aTJ.Phase;
    const double* Freq  = aTJ.Freq;
    for (int aK=0 ; aK<Nb ; aK++)
    {
       Tmp[aK] = Freq[aK]*Square(aParam[aK]+0.2)+Phase[aK];
       if (aK==2)
          Som += cos(Tmp[aK]);
       else
          Som -= cos(Tmp[aK]);
    }
    Som = 1/Som;
    double V2 = Square(Som);
    Eigen::Matrix<double, 1, Nb> mEps;

    for (int aK=0 ; aK<Nb ; aK++)
    {
       // double aP2= aParam[aK]+0.2;
       mEps[aK] = 2*(aParam[aK]+0.2)*Freq[aK] *sin(Tmp[aK]) *V2;
       if (aK==2)
          mEps[aK] *= -1;
    }

    return cEpsNum<Nb>(Som,mEps);
}





template <class Type,int Nb> cTestJets<Type,Nb>::cTestJets() 
{
   for (int aK=0 ; aK<Nb ; aK++)
   {
      Phase[aK]  = aK/2.7 +  1.0 /(1 + aK);
      Freq[aK]  = 1 + aK;
   }
}


#define TJ(K) (COS(Freq[K]*Square(Param[K]+0.2) + Phase[K]))

template <class Type,int Nb> Type cTestJets<Type,Nb>::ComputeExpansed(Type * Param) const
{
    if (Nb==2)
        return   Type(1)/ (TJ(0)+TJ(1));
    if (Nb==6)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5));
    if (Nb==12)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11));

    if (Nb==18)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11)+TJ(12)+TJ(13)+TJ(14)+TJ(15)+TJ(16)+TJ(17));


    if (Nb==24)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11)+TJ(12)+TJ(13)+TJ(14)+TJ(15)+TJ(16)+TJ(17)+TJ(18)+TJ(19)+TJ(20)+TJ(21)+TJ(22)+TJ(23));

/*
    MMVII_INTERNAL_ASSERT_always(false,"COMMPUUTE");
    return   Type(1)/ TJ(0);
*/
    return ComputeLoop(Param);
}

template <class Type,int Nb> Type cTestJets<Type,Nb>::ComputeLoop(Type * Param) const
{
    Type aRes = TJ(0);
    for (int aK=1 ; aK<Nb ; aK++)
    {
        if (aK==2)
           aRes = aRes -TJ(aK);
        else 
           aRes = aRes +TJ(aK);
    }
    return Type(1.0) / aRes;
}

// class cTestJets<double,6>;
// class cTestJets<cEpsNum<6>,6 >;

/*
Eigen::Matrix<double, 1, N> ApplyOp(const Eigen::Matrix<double, 1, N> aMat,OpDouble anOp);
template<int N> cEpsNum<N> operator/(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum / g.mNum, f.mEps * g.mNum + f.mNum * g.mEps);
}
*/

extern bool NEVER;

template <int Nb>  void TplTestJet(double aTiny)
{
   cEpsNum<Nb>  TabJets[Nb];
   double      TabDouble[Nb];
   // On initialise le Jet et les double a la meme valeur
   for (int aK=0 ; aK<Nb ; aK++)
   {
       TabDouble[aK] = tan(aK);
       TabJets[aK] = cEpsNum<Nb>(TabDouble[aK],aK);
   }
   cTestJets<cEpsNum<Nb>,Nb > aTestJet;
   cTestJets<double,Nb >      aTestDouble;
   
   cEpsNum<Nb> aDerAn =  TJ_DerAnalytique(aTestDouble,TabDouble);
   cEpsNum<Nb>  aJetDer = aTestJet.ComputeExpansed(TabJets);
   cEpsNum<Nb>  aJetLoop = aTestJet.ComputeLoop(TabJets);
   for (int aKv=0 ; aKv<Nb ; aKv++)
   {
       double aDerJ = aJetDer.mEps[aKv];
       double  aTdPlus[Nb];
       double  aTdMoins[Nb];
       for (int aKd=0 ; aKd<Nb ; aKd++)
       {
            double aDif   = (aKd==aKv) ? aTiny : 0.0;
            aTdPlus[aKd]  = TabDouble[aKd] + aDif;
            aTdMoins[aKd] = TabDouble[aKd] - aDif;
       }
       double aVPlus =  aTestDouble.ComputeExpansed(aTdPlus);
       double aVMoins = aTestDouble.ComputeExpansed(aTdMoins);
       double aDerNum = (aVPlus-aVMoins) / (2.0 * aTiny);
       double aDif1 = RelativeDifference(aDerJ,aDerNum,nullptr);
       double aDif2 = RelativeDifference(aDerJ,aDerAn.mEps[aKv],nullptr);
       double aDif3 = RelativeDifference(aDerJ,aJetLoop.mEps[aKv],nullptr);
       // StdOut() << "Der; Jet/num=" << aDif1 << " Jet/An=" << aDif2 << " Jet/Loop=" << aDif3 << "\n";
       //  StdOut() << "Der; Jet/num=" << aDerJ << " Jet/An=" << aDerAn.mEps[aKv]  << "\n";
       if (Nb<100)
       {
           MMVII_INTERNAL_ASSERT_bench(aDif1<1e-3,"COMMPUUTE");
       }
// StdOut()  << "__22222: "  << aDif2 << "\n";
       MMVII_INTERNAL_ASSERT_bench(aDif2<1e-10,"COMMPUUTE");
       MMVII_INTERNAL_ASSERT_bench(aDif3<1e-10,"COMMPUUTE");
   }

   for (int aK=0 ; aK< 3; aK++)
   {
      int Time=int(1e6/Nb);
      double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0();
      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerJet = aTestJet.ComputeExpansed(TabJets);
          if (NEVER)
             StdOut() << aDerJet.mNum << "\n";
      }
      double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerAn =  TJ_DerAnalytique(aTestDouble,TabDouble);
          IgnoreUnused(aDerAn);
          if (NEVER)
             StdOut() << aDerAn.mNum << "\n";
      }
      double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerLoop = aTestJet.ComputeLoop(TabJets);
          if (NEVER)
             StdOut() << aDerLoop.mNum << "\n";
      }
      double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0();
      
      StdOut() << "TIME; Ratio Jet/An=" << (aT1-aT0) / (aT2-aT1) 
               << " Jets= " << aT1-aT0 
               << " An=" << aT2-aT1 
               << " Loop=" << aT3-aT2 
               << "\n";
   }
   StdOut() << "============== Nb=" << Nb << "\n";
}


template <const int Nb> void  TplBenchDifJets()
{
    {
       static int aCpt=0,aNbTot=0,aNbNot0=0; aCpt++;

       cEpsNum<Nb> anEps1 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps1 = anEps1.ToVEN();
       double aDif1 = EpsDifference(anEps1,aVEps1);
       aNbTot+= Nb;
       aNbNot0 +=  aVEps1.mVInd.size();

       if (aCpt>100)
       {
           double aProp  = aNbNot0/double(aNbTot);
           // Verif que les  proportion sont respectees pour que test soit probants
           MMVII_INTERNAL_ASSERT_bench((aProp>0.25) && (aProp<0.75),"TplBenchDifJets");
       }
       // Verif conversion, 
       MMVII_INTERNAL_ASSERT_bench(aDif1<1e-5,"TplBenchDifJets");


       //  operation algebrique
       cEpsNum<Nb> anEps2 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps2 = anEps2.ToVEN();

       cEpsNum<Nb> anEps3 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps3 = anEps3.ToVEN();
       //
       cEpsNum<Nb> anEpsCmp = 2.7+ 1.2*anEps1*1.3 + anEps2*COS(anEps3)+3.14;
       cVarEpsNum  aVEpsCmp = 2.7+ 1.3*aVEps1*1.2 + aVEps2*COS(aVEps3)+3.14;

       double 	aDifCmp = EpsDifference(anEpsCmp,aVEpsCmp);
       MMVII_INTERNAL_ASSERT_bench(aDifCmp<1e-5,"TplBenchDifJets");
    }
    double aVNum =   RandUnif_C();
    cVarEpsNum  aVE(aVNum);
    cEpsNum<Nb> aE(aVNum);
    int aNbCoeff = RandUnif_N(Nb);
    for (int aTime=0 ; aTime<aNbCoeff ; aTime++)
    {
        int aK =  RandUnif_N(Nb);
        double aVal = RandUnif_C();
        aVE.AddVal(aVal,aK);
        aE.mEps[aK] += aVal;
        if (aTime%2==0)
        {
           aVE.mVEps.push_back(0.0);
           aVE.mVInd.push_back(aK+Nb);  // parfois dehors, parfois dedans, pas d'influence si nul
        }
    }
    MMVII_INTERNAL_ASSERT_bench(EpsDifference(aE,aVE)<1e-5,"TplBenchDifJets");

    double aTheoDif = 0.0;
    double aV0 = RandUnif_C();
    aTheoDif += std::abs(aV0);
    aVE.AddVal(aV0,Nb+2);
    double aD0 = EpsDifference(aE,aVE);
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD0-aTheoDif)<1e-5,"TplBenchDifJets");


    double aV1 = RandUnif_C();
    aE.mEps[1] += aV1;
    aTheoDif += std::abs(aV1);
    double aD1 = EpsDifference(aE,aVE);
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aTheoDif)<1e-5,"TplBenchDifJets");

    for (int aTime=0 ; aTime<3 ; aTime++)
    {
        int aK =  RandUnif_N(Nb);
        double aVal = RandUnif_C();
        aVE.AddVal(aVal,aK);
        aE.mEps[aK] += aVal;
        if (aTime%2==0)
        {
           aVE.mVEps.push_back(0.0);
           aVE.mVInd.push_back(aK+Nb);  // parfois dehors, parfois dedans, pas d'influence si nul
        }
    }
    MMVII_INTERNAL_ASSERT_bench(std::abs(aTheoDif-EpsDifference(aE,aVE))<1e-5,"TplBenchDifJets");
}
/*
template <const int Nb> void  TplOpJets()
{
    cVarEpsNum  aVE(0.0);
    cEpsNum<Nb> aE;
}
*/

void BenchMyJets()
{

   for (int aK=0 ; aK<1000 ; aK++)
   {
      TplBenchDifJets<10>();
      TplBenchDifJets<60>();
   }
   

   //=====

   TplTestJet<2>(1e-5);
   TplTestJet<6>(1e-5);
   TplTestJet<12>(1e-5);
   TplTestJet<18>(1e-5);
   TplTestJet<24>(1e-5);
   TplTestJet<48>(1e-5);
   TplTestJet<96>(1e-5);
   TplTestJet<192>(1e-5);
   TplTestJet<484>(1e-5);

   getchar();
   cEpsNum<20>  aEps(3.0,1);
   cPtxd<cEpsNum<20>,3> aP1(aEps,aEps,aEps);
   cPtxd<cEpsNum<20>,3> aP2;
   cPtxd<cEpsNum<20>,3> aP3 = aP1+aP2;
   
   IgnoreUnused(aP3);

/*
   cEpsNum<255>  aEps(3.0,1);
   aEps = COS(aEps) + aEps*aEps + Square(aEps)/aEps;
   StdOut() << aEps.mEps;
   cProjCamRad<double> aCamD;
*/
   cProjCamRad<cEpsNum<6> > aCamJ6;
   aCamJ6.Compute(nullptr,nullptr);
   // cProjCamRad<cEpsNum<6> >::ParamInitStd();
       // void Compute (Type * Parameter,Type * Residual);

/*
   cEpsNum<6>  TabJ6[6];
   double      TabD6[6];
   // On initialise le Jet et les double a la meme valeur
   for (int aK=0 ; aK<6 ; aK++)
   {
       TabD6[aK] = tan(aK);
       TabJ6[aK] = cEpsNum<6>(TabD6[aK],aK);
   }
   cTestJets<cEpsNum<6>,6 > aTestJ6;
   
   cEpsNum<6>  aJ6Der = aTestJ6.ComputeExpansed(TabJ6);
*/
}
bool NEVER=false;



};
