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



#include "StdAfx.h"

// Classe generique , a rebalancer dans la future BB 
// d'optimisation generale

#include <vector>

extern bool DebugOFPA ;
extern int aCPTOkOFA ;
extern int aCPTNotOkOFA ;


/*
class RacineFonc1D
{
     public :

         void solutions_1prof
              (
                  ElSTDNS vector<REAL> &,
                  ElSTDNS vector<REAL> &,
                  REAL x0,
                  REAL x1,
                  INT  nb_step
                  
              );
  virtual ~RacineFonc1D() {}
     private :

         virtual REAL ValRF1D(REAL x,bool & def) =0;
        
};


void RacineFonc1D::solutions_1prof
     (
         ElSTDNS vector<REAL> & bsup,
         ElSTDNS vector<REAL> & binf,
         REAL X0,
         REAL X1,
         INT  nb_step
     )
{
    bsup.clear();
    binf.clear();

    bool ok0 = true;
    REAL x0 = X0;
    REAL v0 = ValRF1D(x0,ok0);
    for (INT k=1 ; k <= nb_step ; k++)
    {
         bool ok1 = true;
         REAL x1 = (k*X1 + (nb_step-k) *X0) / nb_step;
         REAL v1 = ValRF1D(x1,ok1);

         if (ok0 && ok1 && ((v0>0) != (v1>0)))
         {
            binf.push_back(x0);
            bsup.push_back(x1);
         }
         ok0 = ok1;
         x0 =  x1;
         v0 =  v1;
    }
}
*/


/*
    Nouvelle classe pour (tenter de) resoudre les problemes d'instabilite numerique apparues sur donnees video drone tres longue focale.


    On utilise des polynome en b' c' avec b = 1+ b'   et c= 1+c'

*/

template <class Type>  ElPolynome<Type>  PolSquareScal(Pt3d<Type> aCste, Pt3d<Type> aVar)
{
    return ElPolynome<Type> (square_euclid(aCste),2*scal(aCste,aVar),square_euclid(aVar));
}

template <class Type> class  cNewResProfChamp
{
     public :
        cNewResProfChamp(Type aDistBC,Pt3dr pa,Pt3dr pb,Pt3dr pc,Type aRhoA,Type aRhoC) :

           mDistBC (aDistBC),
           mRhoA (aRhoA),
           mRhoC (aRhoC),
           mPa   (pa.x,pa.y,pa.z),
           mPb   (pb.x,pb.y,pb.z),
           mPc   (pc.x,pc.y,pc.z),

           mVca  (mPa-mPc),
           mVba  (mPa-mPb),
           mVcb  (mPb-mPc),

           mA2   (square_euclid(mPa)),
           mB2   (square_euclid(mPb)),
           mC2   (square_euclid(mPc)),

           mCA2 (square_euclid(mVca)),
           mBA2 (square_euclid(mVba)),

           mScac (scal(mVca,mPc)),
           mSbab (scal(mVba,mPb)),
           mBeta (mSbab / mB2),
           mU    (mVcb +  mPb *mBeta),

           // mDeltaBOfC  (ElSquare(mBeta)+(mRhoA*mCA2-mBA2)/mB2 , (-2*mScac*mRhoA)/mB2, (mRhoA *mC2) / mB2)
           mDeltaBOfC  ( ElPolynome<Type>(ElSquare(mBeta)-mBA2/mB2) + PolSquareScal(mVca,-mPc) *(mRhoA/mB2)),


           mW1         (PolSquareScal(mVca,-mPc)),
           mW2         (PolSquareScal(mU,-mPc)),
           mOmega      (mW1*mRhoC - mW2 - mDeltaBOfC *mB2),
           mAlpha      (scal(mU,mPb),-scal(mPc,mPb)),
           mCSolve     (ElSquare(mOmega) - (mDeltaBOfC*ElSquare(mAlpha)) * 4)
         {
         }

         Type BOfC(Type aC, int aSign,bool & Ok);
         std::vector<Type>  CRoots();
         Type ScoreCandSigne(Type aC, int aSign,bool & Ok,Type & aB);


         Type RatioA(Type aB,Type aC);
         Type RatioC(Type aB,Type aC);
         Type QualSol(Type aB,Type aC);

         Pt3d<Type> FandDerRatA(Type b,Type c);
         Pt3d<Type> FandDerRatC(Type b,Type c);
         void CheckDeriv(Type b,Type c);


         void AffineBC(Type & aB,Type & aC);


         std::list<Pt3dr> GetSols();

     private :

        Type mDistBC;
        Type mRhoA;
        Type mRhoC;

        Pt3d<Type>  mPa;
        Pt3d<Type>  mPb;
        Pt3d<Type>  mPc;
               
        Pt3d<Type>  mVca;  // mPa - mPc  : "Vecteur CA"
        Pt3d<Type>  mVba;  // mPb - mPa
        Pt3d<Type>  mVcb;  // mPb - mPc

        Type        mA2;  // mPb^2
        Type        mB2;  // mPb^2
        Type        mC2;  // mPc^2
        Type        mCA2;  // mVca^2
        Type        mBA2;  // mVba^2

        Type        mScac;  // AC . C
        Type        mSbab;  // AC . C
        Type        mBeta;

        Pt3d<Type>  mU;
        
         // En utilisant RhoA = (mVba -b' mPb)^2  / (mVca-c' mPc) ^2
         //  b'^2  - 2 mBeta b' = 
         // On tire   b' = mBeta +- sqrt(Delta(c'))
         //  avec
         //           mBeta = mSbab / mB2
         //   Delta(c') = mBeta ^2 + [ rhoA * (mCA2 - 2 c mScac + c^2  mC2) - mBA2  ] / mB2

        ElPolynome<Type> mDeltaBOfC;
        ElPolynome<Type> mW1;
        ElPolynome<Type> mW2;
        ElPolynome<Type> mOmega;
        ElPolynome<Type> mAlpha;
        ElPolynome<Type> mCSolve;
/*
*/
     
};

template <class Type> Type  cNewResProfChamp<Type>::BOfC(Type aC, int aSign,bool & Ok)
{
    Type aDC = mDeltaBOfC(aC);
    if (aDC < 0)
    {
      Ok=false;
      return 1e60;
    }

    Ok=true;
    return mBeta + aSign * sqrt(aDC);
}

template <class Type> std::vector<Type>  cNewResProfChamp<Type>::CRoots()
{
   std::vector<Type> aRoots;
   RealRootsOfRealPolynome<Type>(aRoots,mCSolve,Type(1e-16),100);
   return aRoots;
}

                  // ===== Ratio A

template <class Type> Type cNewResProfChamp<Type>::RatioA(Type aB,Type aC)
{
  return square_euclid(mVba-mPb*aB) / square_euclid(mVca-mPc*aC);
}

template <class Type> Pt3d<Type> cNewResProfChamp<Type>::FandDerRatA(Type b,Type c)
{
    Type aNum = square_euclid(mVba-mPb*b) ;
    Type aDen = square_euclid(mVca-mPc*c);

    Type aVal = aNum/aDen;

    Type aDb = 2 * (b*mB2 - scal(mPb,mVba)) / aDen; 
    Type aDc = -2 * (c*mC2 - scal(mPc,mVca))  * (aNum/ElSquare(aDen));


    return Pt3d<Type> (aDb,aDc,aVal);
}

                  // ===== Ratio C


template <class Type> Type cNewResProfChamp<Type>::RatioC(Type aB,Type aC)
{
  return square_euclid(mVcb + mPb*aB - mPc*aC) / square_euclid(mVca-mPc*aC);
}

template <class Type> Pt3d<Type> cNewResProfChamp<Type>::FandDerRatC(Type b,Type c)
{
     Type aNum = square_euclid(mVcb + mPb*b - mPc*c) ;
     Type aDen = square_euclid(mVca-mPc*c);

     Type aVal = aNum/aDen;

     Type aDb = 2 * ( mB2 * b + scal(mPb,mVcb-mPc*c)) / aDen;

     Type aDc = 2*(aDen*(c*mC2-scal(mPc,mVcb+mPb*b))-aNum*(c*mC2-scal(mPc,mVca) ))  /   ElSquare(aDen);


    return Pt3d<Type> (aDb,aDc,aVal);

}

template <class Type>  void  cNewResProfChamp<Type>::CheckDeriv(Type  b,Type  c)
{
    Type aEps = 1e-8;

    Pt3d<Type> aDA = FandDerRatA(b,c);
    Pt3d<Type> aDC = FandDerRatC(b,c);


    Type A0 = RatioA(b,c);
    Type C0 = RatioC(b,c);

    Type Abp1 = RatioA(b+aEps,c);
    Type Abm1 = RatioA(b-aEps,c);
    Type DaDb = (Abp1-Abm1) / (2*aEps) ;
    Type Acp1 = RatioA(b,c+aEps);
    Type Acm1 = RatioA(b,c-aEps);
    Type DaDc = (Acp1-Acm1) / (2*aEps) ;


    Type Cbp1 = RatioC(b+aEps,c);
    Type Cbm1 = RatioC(b-aEps,c);
    Type DcDb = (Cbp1-Cbm1) / (2*aEps) ;
    Type Ccp1 = RatioC(b,c+aEps);
    Type Ccm1 = RatioC(b,c-aEps);
    Type DcDc = (Ccp1-Ccm1) / (2*aEps) ;



    std::cout << "Chekc Vals "<< ElAbs(aDA.z-A0) << " " << ElAbs(aDC.z-C0) << "\n";
    std::cout << "  DerA :: "<< ElAbs(aDA.x-DaDb) << " " << ElAbs(aDA.y-DaDc) << "\n";
    std::cout << "  DerC :: "<< ElAbs(aDC.x-DcDb) << " " << ElAbs(aDC.y-DcDc) << "\n";
}
    



template <class Type> Type cNewResProfChamp<Type>::QualSol(Type aB,Type aC)
{
   return ElAbs(RatioA(aB,aC)-mRhoA) + ElAbs(RatioC(aB,aC)-mRhoC);
}



template <class Type> Type  cNewResProfChamp<Type>::ScoreCandSigne(Type aC, int aSign,bool & Ok,Type & aB)
{
     aB = BOfC(aC,aSign,Ok);
     if (Ok)
        return QualSol(aB,aC);
     return 1e20;
}


template <class Type>  void  cNewResProfChamp<Type>::AffineBC(Type & b,Type & c)
{
    double aTarget = 1e-10;
    int aNbIterMax = 10;

    bool Cont =true;

    while (Cont)
    {
        Pt3d<Type> aDA = FandDerRatA(b,c);
        Pt3d<Type> aDC = FandDerRatC(b,c);
        Pt2d<Type> aDif (mRhoA-aDA.z,mRhoC-aDC.z);
        Type aEr = ElAbs(aDif.x) + ElAbs(aDif.y);

        if (aEr< aTarget)
        {
            Cont = false;
        }
        else
        {
/*
    A C   D -C   ;  DA.z +   [DA.x    DA.y] * X = rhoA
    B D   -B A   ;  DC.z +   [DC.x    DC.y]   Y = rhoC
*/
            Type aDelta = aDA.x * aDC.y -aDC.x*aDA.y;
            if (ElAbs(aDelta) > 1e-20) 
            {
                 Pt2d<Type> aInvGx (  aDC.y/aDelta, -aDA.y/aDelta) ;
                 Pt2d<Type> aInvGy ( -aDC.x/aDelta,  aDA.x/aDelta) ;

                 Type aNewB = b + scal(aInvGx,aDif);
                 Type aNewC = c + scal(aInvGy,aDif);

                 if (QualSol(aNewB,aNewC) < aEr)
                 {
                     b = aNewB;
                     c = aNewC;
                 }
                 else
                 {
                     Cont = false;
                 }
            }
            else
            {
                Cont = false;
            }
        }
        aNbIterMax--;
        if (aNbIterMax<=0) 
        {
            Cont = false;
        }
    }
}




template <class Type>  std::list<Pt3dr> cNewResProfChamp<Type>::GetSols()
{
     std::list<Pt3dr> aRes;
     std::vector<Type> aVC = CRoots();

     for (int aK=0 ; aK<int(aVC.size()) ; aK++)
     {
          Type aCostMin = 1e10;
          Type aBestC=0;
          Type aBestB=0;
          for (int aS=-1 ; aS<=1 ; aS+=2)
          {
              bool Ok;
              Type aB;
              Type aSc = ScoreCandSigne(aVC[aK],aS,Ok,aB);

              if (Ok && (aSc<aCostMin))
              {
                 aCostMin = aSc;
                 aBestC= aVC[aK];
                 aBestB= aB;
              }
          }

          if (aCostMin < 1e-2)
          {
              AffineBC(aBestB,aBestC);
              if (QualSol(aBestB,aBestC) < 1e-6)
              {
                  aBestC += 1;
                  aBestB += 1;
                  Type ratio = mDistBC / euclid(mPb*aBestB-mPc*aBestC);
                  aRes.push_back(Pt3dr(1,aBestB,aBestC)*ratio);
              }
          }
     }

     return aRes;
}



/*  
                   ===============  ANCIENNE =====================
*/

template <class Type> class  ResProfChamp 
{
     public :
        ResProfChamp(Type DistBC,Pt3dr pa,Pt3dr pb,Pt3dr pc,Type rhoA,Type rhoC);


        void ListeC(ElSTDNS list<Pt3dr>&  res);

     private :

        void AddBC(std::list<Pt3dr> & res,Type b,Type c,bool FromApproxLin);
        bool OkSolC(Type c);

        Type BfromC(Type c,bool & OK);
        // virtual Type ValRF1D(Type x,bool & def);

        Type RatA (Type b,Type c);
        Type RatC (Type b,Type c);
        
        Pt3d<Type> FandDerRatA (Type b,Type c);
        Pt3d<Type> FandDerRatC (Type b,Type c);
        Type Error (Type b,Type c);

        void CheckDerivatives(Type & b,Type & c);
        void Optimize(Type & b,Type & c);

        Type   _DistBC;
        Pt3d<Type>  _PA;
        Pt3d<Type>  _PB;
        Pt3d<Type>  _PC;

        Type   _A2;
        Type   _B2;
        Type   mC2;
        Type   _AB;
        Type   _AC;
        Type   _BC;

        Type   _rhoA;
        Type   _rhoC;

        Type   _g;
        Type   _gam0;
        Type   _gam1;
        Type   _gam2;
        ElPolynome<Type> _Gamma;

        Type             _omega0;
        Type             _omega1;
        Type             _omega2;
        ElPolynome<Type> _Omega;
        ElPolynome<Type> _Alpha;
        ElPolynome<Type> _Resolv;
        ElSTDNS vector<Type>     mRoots;

        INT    _signdisc;

};

template <class Type> Type  ResProfChamp<Type>::RatA(Type b,Type c)
{
    return square_euclid(_PA-_PB*b) / square_euclid(_PA-_PC*c);
}

template <class Type> Type   ResProfChamp<Type>::RatC(Type b,Type c)
{
     return square_euclid(_PB*b-_PC*c) / square_euclid(_PA-_PC*c);
}


template <class Type> Pt3d<Type>  ResProfChamp<Type>::FandDerRatA(Type b,Type c)
{
    Type aNum =  square_euclid(_PA-_PB*b) ;
    Type aDen =  square_euclid(_PA-_PC*c);

    Type aVal = aNum/aDen;

    Type aDb = 2* (_B2*b - _AB) / aDen;

    Type aDc =  -(2*aNum *(mC2*c-_AC)) /ElSquare(aDen) ;

     return Pt3d<Type> (aDb,aDc,aVal);
}

template <class Type> Pt3d<Type>  ResProfChamp<Type>::FandDerRatC(Type b,Type c)
{
    Type aNum =  square_euclid(_PB*b-_PC*c);
    Type aDen =  square_euclid(_PA-_PC*c);

    Type aVal = aNum/aDen;

    Type aDb = 2* (_B2*b - c* _BC) / aDen;

    Type aDc =  2 * (aDen *(mC2*c -b*_BC)  -  aNum *(mC2*c-_AC)) /ElSquare(aDen) ;

     return Pt3d<Type> (aDb,aDc,aVal);
}
/*
*/

template <class Type>  void  ResProfChamp<Type>::CheckDerivatives(Type & b,Type & c)
{
    Type aEps = 1e-6;

    Pt3d<Type> aDA = FandDerRatA(b,c);
    Pt3d<Type> aDC = FandDerRatC(b,c);


    Type A0 = RatA(b,c);
    Type C0 = RatC(b,c);

    Type Abp1 = RatA(b+aEps,c);
    Type Abm1 = RatA(b-aEps,c);
    Type DaDb = (Abp1-Abm1) / (2*aEps) ;
    Type Acp1 = RatA(b,c+aEps);
    Type Acm1 = RatA(b,c-aEps);
    Type DaDc = (Acp1-Acm1) / (2*aEps) ;


    Type Cbp1 = RatC(b+aEps,c);
    Type Cbm1 = RatC(b-aEps,c);
    Type DcDb = (Cbp1-Cbm1) / (2*aEps) ;
    Type Ccp1 = RatC(b,c+aEps);
    Type Ccm1 = RatC(b,c-aEps);
    Type DcDc = (Ccp1-Ccm1) / (2*aEps) ;



    std::cout << "ptimizeBC "<< ElAbs(aDA.z-A0) << " " << ElAbs(aDC.z-C0) << "\n";
    std::cout << "  DerA :: "<< ElAbs(aDA.x-DaDb) << " " << ElAbs(aDA.y-DaDc) << "\n";
    std::cout << "  DerC :: "<< ElAbs(aDC.x-DcDb) << " " << ElAbs(aDC.y-DcDc) << "\n";
    
}

template <class Type>  Type  ResProfChamp<Type>::Error (Type b,Type c)
{
   return ElAbs(RatA(b,c)-_rhoA) +  ElAbs(RatC(b,c)-_rhoC);
}


template <class Type>  void  ResProfChamp<Type>::Optimize(Type & b,Type & c)
{
    std::cout << "INPUT ERR " << Error(b,c)  << " b-1=" << (b-1) << " c-1=" << (c-1) << "\n";
    double aTarget = 1e-6;
    int aNbIterMax = 20;

    bool Cont =true;
    std::string aMesOut="";

    while (Cont)
    {
        Pt3d<Type> aDA = FandDerRatA(b,c);
        Pt3d<Type> aDC = FandDerRatC(b,c);
        Pt2d<Type> aDif (_rhoA-aDA.z,_rhoC-aDC.z);
        Type aEr = ElAbs(aDif.x) + ElAbs(aDif.y);

        if (aEr< aTarget)
        {
            Cont = false;
            aMesOut="Got!!";
        }
        else
        {
/*
    A C   D -C   ;  DA.z +   [DA.x    DA.y] * X = rhoA
    B D   -B A   ;  DC.z +   [DC.x    DC.y]   Y = rhoC
*/
            Type aDelta = aDA.x * aDC.y -aDC.x*aDA.y;
            if (ElAbs(aDelta) > 1e-20) 
            {
                 Pt2d<Type> aInvGx (  aDC.y/aDelta, -aDA.y/aDelta) ;
                 Pt2d<Type> aInvGy ( -aDC.x/aDelta,  aDA.x/aDelta) ;

                 Type aNewB = b + scal(aInvGx,aDif);
                 Type aNewC = c + scal(aInvGy,aDif);

                 if (Error(aNewB,aNewC) < aEr)
                 {
                     b = aNewB;
                     c = aNewC;
                 }
                 else
                 {
                     aMesOut="Grow";

/*
std::cout << "ERR0 " << aEr << "\n";
for (int aK=0 ; aK<5 ; aK++)
{
   std::cout  << "   ErK " << Error(b+scal(aInvGx,aDif)/pow(2,aK), c + scal(aInvGy,aDif)/pow(2,aK)) << "\n";
}
*/
                     Cont = false;
                 }
            }
            else
            {
                aMesOut="Delta";
                Cont = false;
            }
        }
        aNbIterMax--;
        if (aNbIterMax<=0) 
        {
            aMesOut="NbIter";
            Cont = false;
        }
    }

    // std::cout << "OUT ERR " << Error(b,c)  << " ITRED " << aNbIterMax  << " Reason=" << aMesOut << " " << (b-1) << " " << (c-1) << "\n";

}



template <class Type> ResProfChamp<Type>::ResProfChamp
(
     Type  DistBC,
     Pt3dr pa,
     Pt3dr pb,
     Pt3dr pc,
     Type rhoA,
     Type rhoC
)  :
   _DistBC      (DistBC),
   _PA          (pa.x,pa.y,pa.z),
   _PB          (pb.x,pb.y,pb.z),
   _PC          (pc.x,pc.y,pc.z),
   _A2          (square_euclid(_PA)),
   _B2          (square_euclid(_PB)),
   mC2          (square_euclid(_PC)),
   _AB          (scal(_PA,_PB)),
   _AC          (scal(_PA,_PC)),
   _BC          (scal(_PB,_PC)),

   _rhoA        (rhoA),
   _rhoC        (rhoC),

   _g           (_AB/_B2),
   _gam0        (ElSquare(_AB/_B2)+(_rhoA-1)*(_A2/_B2)),
   _gam1        (-2*_rhoA * (_AC/_B2)),
   _gam2        (_rhoA * mC2 /_B2),
   _Gamma       (_gam0,_gam1,_gam2),

   _omega0      (_rhoC*_A2 -_g*_g*_B2 -_B2*_gam0),
   _omega1      (2*(_g*_BC-_rhoC*_AC)-_B2*_gam1),
   _omega2      ((_rhoC-1)*mC2 -_B2*_gam2),
   _Omega       (_omega0,_omega1,_omega2),

   _Alpha       (_B2*_g,-_BC),
   _Resolv      (_Omega *_Omega -_Gamma*_Alpha*_Alpha *4.0),

   _signdisc    (1)

{
/*
if ( DebugOFPA)
std::cout << "ResProfChamp:: " << (pa.z-1) << " " << (pb.z-1) << " " << (pc.z-1) << "\n";
*/
}



template <class Type> Type ResProfChamp<Type>::BfromC(Type c,bool & OK)
{
     Type Gamma = _Gamma(c);


     OK = (Gamma >=0);
     if (! OK) 
        return 1e60;

     return _g + _signdisc * sqrt(Gamma);
}

template <class Type> bool ResProfChamp<Type>::OkSolC(Type c)
{
     Type Gamma = _Gamma(c);


     if (Gamma <0)
        return false;

     if (ElAbs(_Omega(c) - 2 * _signdisc * sqrt(Gamma) * (_B2*_g -_BC*c))> 1e-4)
        return false;

     return true;
}


template <class Type> void ResProfChamp<Type>::AddBC(std::list<Pt3dr> & res,Type b,Type c,bool FromApproxLin)
{
    // Optimize(b,c);

/*
    if (0)
    {
       std::cout << " BBB-CHECK ";
       for (int aSign=-1; aSign<=1 ; aSign+=2)
       {
            bool Ok;
            Type aNewB = 1+mNRP.BOfC(c-1,aSign,Ok);
            if (Ok) std::cout << ElAbs(aNewB-b) << " ";
       }
       std::cout << "\n";

       std::cout << " CCC-CHECK ";
       std::vector<Type> aVC = mNRP.CRoots();
       for (int aK=0 ; aK<int(aVC.size()) ; aK++)
           std::cout << ElAbs(aVC[aK]+1-c) << " ";
       std::cout << "\n";
 
    }
*/

    Type aRatA = RatA(b,c);
    Type aRatC = RatC(b,c);
    if ( 
             (ElAbs(aRatA-_rhoA) < 1e-4)
         &&  (ElAbs(aRatC-_rhoC) < 1e-4)
    )
    {
        Type ratio = _DistBC / euclid(_PB*b-_PC*c);
        res.push_back(Pt3dr(1,b,c)*ratio);
    }
    else
    {
        if (DebugOFPA)
        {
/*
                         std::cout << "BEGINDbgOFPA \n";
                         std::cout << ElAbs(RatA-_rhoA) << " " << ElAbs(RatC-_rhoC) << "\n";
                         std::cout << " POL " <<  c << " " << _Resolv(c) << "\n";

*/
        }
    }
}

template <class Type> void ResProfChamp<Type>::ListeC(ElSTDNS list<Pt3dr>&  res)
{
     // mNRP.GetSols();
     res.clear();

     // ELISE_ASSERT(false,"REMETTRE ResProfChamp<Type>::ListeC");
     RealRootsOfRealPolynome<Type>(mRoots,_Resolv,Type(1e-10),100);

     int OKS = 0;

     for (_signdisc =-1; _signdisc <=1 ; _signdisc+=2)
     {
         for (INT k=0; k<(INT) mRoots.size(); k++) 
         {
            Type c = mRoots[k];
            if (OkSolC(c))
            {
               OKS ++;
               bool OK;
               Type b = BfromC(c,OK);
               ELISE_ASSERT(OK,"Incoh in ResProfChamp::ListeC");
               AddBC(res,b,c,false);

/*

               Optimize(b,c);
               Type aRatA = RatA(b,c);
               Type aRatC = RatC(b,c);
               if ( 
                        (ElAbs(aRatA-_rhoA) < 1e-4)
                    &&  (ElAbs(aRatC-_rhoC) < 1e-4)
                  )
               {
                   Type ratio = _DistBC / euclid(_PB*b-_PC*c);
                   res.push_back(Pt3dr(1,b,c)*ratio);
if (DebugOFPA) aCPTOkOFA++;
               }
               else
               {
                     if (DebugOFPA)
                     {
if (DebugOFPA) aCPTNotOkOFA++;
                     }
               }
*/
            }
         }
     }
}


Pt3dr   ElPhotogram::PProj(Pt2dr p)
{
    return Pt3dr(p.x,p.y,1);
}

void ShowSol(const std::string & aMes, const std::list<Pt3dr>  & aSol)
{
    std::cout << aMes;
    for (std::list<Pt3dr>::const_iterator itS = aSol.begin() ; itS!= aSol.end() ; itS++)
        std::cout << * itS << " ";
    std::cout << "\n";
}

void ElPhotogram::ProfChampsFromDist
     (
                 ElSTDNS list<Pt3dr>&  res,  // liste de triplets de prof de champs
                 Pt3dr A,Pt3dr B,Pt3dr C, // points de projection
                 REAL dAB, REAL dAC, REAL dBC
     )
{
    if (0)
    {
       std::list<Pt3dr> aOldSol;
       ResProfChamp<REAL16>  RPC(dBC,A,B,C,ElSquare(dAB/dAC),ElSquare(dBC/dAC));
       RPC.ListeC(aOldSol);
       ShowSol("Old : ",aOldSol);
     }



    cNewResProfChamp<REAL16>  aNewRPC(dBC,A,B,C,ElSquare(dAB/dAC),ElSquare(dBC/dAC));
    res = aNewRPC.GetSols();
/*
    ShowSol("New : ",res);
    getchar();
*/

// getchar();
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
