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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"
#include "GpGpu/GBV2_ProgDynOptimiseur.h"


class cQckInterpolEpip;
class cCensusGr;
class cOnePCGr;
template <class Type> class cFlagTabule;
template <class Type> class cFlagPonder;
template <class Type> class cImFlags ;




// Redifinition de la moitie superieure du graphe de 8 voisins
//
//     V3  V2 V1
//       \ | / 
//         X -  V0

static int VX[4] = {1,1,0,-1};
static int VY[4] = {0,1,1,1};

// Structure optimisee lorsque l'on veut interpoler 
// une image plusieur fois avec le meme X
class cQckInterpolEpip
{
    public :
        cQckInterpolEpip(float X) :
            mX0     (round_down(X)) ,
            mPds1   (X-mX0),
            mPds0   (1-mPds1)
        {
        } 

        int mX0;
        float mPds1;
        float mPds0;

         inline double  GetVal(float *aV)
         {
              return mPds0 * aV[0] + mPds1 * aV[1];
         }

         inline double  GetVal(float **aV, Pt2di aP)
         {
              float * aL = aV[aP.y] + aP.x + mX0;
              return mPds0 * aL[0] + mPds1 * aL[1];
         }
        
};

class cSomGr
{
     public :
        cSomGr(const Pt2di & aV,const int & aP,const int & aNum) :
             mV   (aV),
             mP   (aP),
             mNum (aNum),
             mFlag(1<<aNum)
        {
        }

       Pt2di mV;
       int   mP;
       int   mNum;
       int   mFlag;

};

class cOnePCGr
{
    friend class cCensusGr;
    public :
        cOnePCGr (const Pt2di & aP0) :
            mP0 (aP0)
        {
        }
        void AddV(const Pt2di & aV,const int & aP,int aNum)
        {
            mV.push_back(cSomGr(aV,aP,aNum));
        }

    private :
        Pt2di mP0;
        std::vector<cSomGr>  mV;
};

class cCensusGr
{
    public :
       // FactDepond =>  0 aucune depond  ; 1 => le pds theo est 1 pour 00 et 0 pour 
       cCensusGr(const Pt2di & aV,const double & FactDepond,bool DoFlag,cCensusGr * aGrShareFlag = 0);
       double GainBasic(float ** Im1,float ** Im2,int  aPx2);
       double CostFlag(int aFlag)
       {
            return mDCF[aFlag] / double(mSomP);
       }

       Im2D_INT4 CalcFlag(Im2D_REAL4 anIm);

    private :
       int FlagVois(float ** Im);
       cCensusGr(const cCensusGr&);
       double                mFactDepond;
       std::vector<cOnePCGr> mPCGR; 
       int                   mSomP;
       Pt2di                 mSzV;
       int                   mNbSom;

       static const int     mMulPds = 1000;
       int                  mNbFlag;
       Im1D_INT4            mCostFlag;
       INT4 *               mDCF;
};

template <class Type> class cBufOnImage
{
    public :
        typedef Type                            tType;
        typedef typename El_CTypeTraits<Type>::tBase     tBase;

        cBufOnImage(Type **,Box2di aBoxDef,Box2di aBoxCalc);

        static cBufOnImage<Type> * FullBufOnIm(Im2D<tType,tBase> anIm);
        
        Type ** data() {return mData;}
        void AvanceX();
        void AvanceY() {mData++;}
    private :
        cBufOnImage(const cBufOnImage<Type> &) ; // N.I.

        void InitY();
        Box2di mBoxDef;
        Box2di mBoxCalc;
        Box2di mBoxTot;
        std::vector<Type *> mVData;
        Type **             mData;
        Type **             mData0;
        int                 mNbL;
        int                 mRabY;
        
};


template <class Type> class cFlagTabule
{
     public :
           typedef Type                            tType;
           typedef typename El_CTypeTraits<Type>::tBase     tBase;
           typedef Im2D<Type,tBase>                tIm2D;
           typedef Im1D<Type,tBase>                tIm1D;
           static const int TheNbBits = 8 *  El_CTypeTraits<Type>::eSizeOf;
           static const int TheFlagLim  = 1<< TheNbBits;

           cFlagTabule(int aNbFlag);

     
     protected :
           int     mNbFlag;
           int     mNbEl;
};


// std::vector<double> 

template <class Type> class cFlagPonder : cFlagTabule<Type>
{
    public :
         static cFlagPonder<Type> * PonderSomFlag(const std::vector<double> & aV,const Pt2di & aVois=Pt2di(-1,-1));
         static cFlagPonder<Type>  * PonderSomFlag(const Pt2di & aV,double Depond,double aGama=1.0);

         double ValPonder(Type *);
         double ValPonderDif(Type *,Type *);
         Pt2di Vois();

    private :
          cFlagPonder(int aNbBits,const Pt2di & aVois=Pt2di(-1,-1));
          cFlagPonder(const cFlagPonder<Type> &); // N.I.

          std::vector<int>          mVNbB;
          std::vector<int>          mVSz;
          std::vector<Im1D_REAL4>   mVIm1D;
          std::vector<float *>      mDIm;
          float **                  mData;
          Pt2di                     mVois;
          bool                      mHasVois;
};

std::vector<double> PondVois(const Pt2di & aV,double Depond,double aGama=1.0); // Si Depond = 0, tous egaux
int NbSomOfVois(const Pt2di & aVois);


template <class Type> class cImFlags : public cFlagTabule<Type>
{
     public :
           typedef Type                            tType;
           typedef typename El_CTypeTraits<Type>::tBase     tBase;
           typedef Im2D<Type,tBase>                tIm2D;
           static const int TheNbBits = 8 *  El_CTypeTraits<Type>::eSizeOf;
           static const int TheFlagLim  = 1<< TheNbBits;

           static cImFlags<Type>  Census(Im2D_REAL4 anIm,Pt2di aSzV);


           static cImFlags<Type>  CensusMS
                                  (
                                       const std::vector<Im2D_REAL4> & aVIm,
                                       Pt2di aSzVMax,
                                       const std::vector<Pt2di> &aVSz,
                                       const std::vector<double> &aVScale
                                  );


           cImFlags(Pt2di aSz,int aNbFlag);
           // static cImFlags(Pt2di aSz,int aNbFlag);


           Type * Flag(const Pt2di & aP) { return mData[aP.y] + aP.x * cFlagTabule<Type>::mNbEl;}
           void  Init(const Pt2di & aP)
           {
               mDF = Flag(aP);
               mCurF = 1;
           }
           void Next()
           {
                mCurF <<= 1;
                if (mCurF==TheFlagLim)
                {
                      mCurF = 1;
                      mDF++;
                }
           }
           void AddCurFlag() {*mDF |= mCurF;}
           const Pt2di & SzIm() {return mSzIm;}

     private :
           
           Pt2di   mSzIm;
           int     mTxF;
           Pt2di   mSzFlag;
           tIm2D   mIm;

           Type  ** mData;
           Type  *  mDF;
           int      mCurF;
};




      /*********************************************************************/
      /*                                                                   */
      /*                           ::                                      */
      /*                                                                   */
      /*********************************************************************/

int NbSomOfVois(const Pt2di & aVois) {return (1+2*aVois.x)*(1+2*aVois.y);}


std::vector<double> NormalizeSom1(const std::vector<double> & aV)
{
    double aSom = 0.0;
    for (int aK=0 ; aK<int(aV.size()) ; aK++)
        aSom += aV[aK];

    std::vector<double> aRes;
    for (int aK=0 ; aK<int(aV.size()) ; aK++)
        aRes.push_back(aV[aK]/aSom);
    return aRes;
}


std::vector<double> PondVois(const Pt2di & aV,double Depond,double aGama)
{
    std::vector<double> aRes;
    double aN0 = euclid(aV);
    for (int anY=-aV.y ; anY<=aV.y ; anY++)
    {
        for (int anX=-aV.x ; anX<=aV.x ; anX++)
        {
             double aN = euclid(Pt2di(anX,anY));
             double aPds = pow(ElMax(1.0 - aN/aN0,0.0),aGama);
             aPds = (1-Depond) + Depond * aPds;
             aRes.push_back(aPds);
        }
    }
    return aRes;
}

std::vector<Pt3di>  VecKImGen
                  ( 
                       Pt2di aSzVMax,
                       const std::vector<Pt2di> &aVSz,
                       const std::vector<double> &aVSigma
                  )
{
    std::vector<Pt3di> aRes;
    int aNbScale = (int)aVSz.size();

    for (int aKS=1 ; aKS<int(aVSigma.size()) ; aKS++)
    {
         ELISE_ASSERT(aVSigma[aKS-1] < aVSigma[aKS],"Sigma Ordre");
    }

    for (int anY=-aSzVMax.y ; anY<=aSzVMax.y ; anY++)
    {
        for (int anX=-aSzVMax.x ; anX<=aSzVMax.x ; anX++)
        {
            double aScaleMin = 1e6;
            int aKSMin = -1;
            for (int aKS=0 ; aKS<aNbScale ; aKS++)
            {
                Pt2di aSzV = aVSz[aKS];

                if ( (ElAbs(anX)<=aSzV.x) &&  (ElAbs(anY)<=aSzV.y) )
                {
                    double aScale = aVSigma[aKS];
                    if (aScale < aScaleMin)
                    {
                        aScaleMin = aScale;
                        aKSMin = aKS;
                    }
                }
            }
            ELISE_ASSERT(aKSMin>=0,"CensusMS no KS");
            aRes.push_back(Pt3di(anX,anY,aKSMin));

        }
    }
    if (MPD_MM())
    {
/*
           std::cout << aRes << "\n";
           getchar();
[[-4,-3,2],[-3,-3,2],[-2,-3,2],[-1,-3,2],[0,-3,2],[1,-3,2],[2,-3,2],[3,-3,2],[4,-3,2],[-4,-2,2],[-3,-2,2],[-2,-2,1],[-1,-2,1],[0,-2,1],[1,-2,1],[2,-2,1],[3,-2,2],[4,-2,2],[-4,-1,2],[-3,-1,2],[-2,-1,1],[-1,-1,0],[0,-1,0],[1,-1,0],[2,-1,1],[3,-1,2],[4,-1,2],[-4,0,2],[-3,0,2],[-2,0,1],[-1,0,0],[0,0,0],[1,0,0],[2,0,1],[3,0,2],[4,0,2],[-4,1,2],[-3,1,2],[-2,1,1],[-1,1,0],[0,1,0],[1,1,0],[2,1,1],[3,1,2],[4,1,2],[-4,2,2],[-3,2,2],[-2,2,1],[-1,2,1],[0,2,1],[1,2,1],[2,2,1],[3,2,2],[4,2,2],[-4,3,2],[-3,3,2],[-2,3,2],[-1,3,2],[0,3,2],[1,3,2],[2,3,2],[3,3,2],[4,3,2]]
*/
    }
    return aRes;
}

std::vector<std::vector<Pt2di> > VecKImSplit (
                       Pt2di aSzVMax,
                       const std::vector<Pt2di> &aVSz,
                       const std::vector<double> &aVSigma
                  )
{
    std::vector<Pt3di>  aVP = VecKImGen(aSzVMax,aVSz,aVSigma);
    std::vector<std::vector<Pt2di> > aRes(aVSz.size());

    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
         Pt3di aP = aVP[aK];
         aRes[aP.z].push_back(Pt2di(aP.x,aP.y));
    }

    return aRes;
}

std::vector<int>  VecKIm
                  ( 
                       Pt2di aSzVMax,
                       const std::vector<Pt2di> &aVSz,
                       const std::vector<double> &aVSigma
                  )
{
    std::vector<Pt3di>  aVP = VecKImGen(aSzVMax,aVSz,aVSigma);
    std::vector<int> aVKIm;

    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
       aVKIm.push_back(aVP[aK].z);
    return aVKIm;
}

      /*********************************************************************/
      /*                                                                   */
      /*                         cFlagTabule<Type>                         */
      /*                                                                   */
      /*********************************************************************/

template   <class Type> cFlagTabule<Type>::cFlagTabule(int aNbFlag) :
     mNbFlag  (aNbFlag),
     mNbEl    ((aNbFlag+TheNbBits-1)/TheNbBits)
{
   // std::cout << "cFlagTabule::  " << aNbFlag  << " " << TheNbBits << " " << mNbEl << "\n"; getchar();
}

      /*********************************************************************/
      /*                                                                   */
      /*                         cFlagPonder<Type>                         */
      /*                                                                   */
      /*********************************************************************/

template  <class Type> double cFlagPonder<Type>::ValPonder(Type * aTabF)
{
    double aRes = 0;
    for (int aK=0 ; aK< cFlagTabule<Type>::mNbEl ; aK++)
        aRes += mData[aK][aTabF[aK]];
    return aRes;
}
template  <class Type> double cFlagPonder<Type>::ValPonderDif(Type * aTabF1,Type * aTabF2)
{
    double aRes = 0;
    for (int aK=0 ; aK< cFlagTabule<Type>::mNbEl ; aK++)
    {
        aRes += mData[aK][aTabF1[aK] ^ aTabF2[aK]];
    }
    return aRes;
}


template  <class Type> cFlagPonder<Type>::cFlagPonder(int aNbFlag,const Pt2di & aVois) :
         cFlagTabule<Type> (aNbFlag),
         mVois             (aVois),
         mHasVois          (mVois.x>0)
{
    if (mHasVois)  
    {
        ELISE_ASSERT(aNbFlag==NbSomOfVois(aVois),"Incoherence in cFlagPonder");
    }
  
     for (int aK=0 ; aK<aNbFlag ; aK+= cFlagTabule<Type>::TheNbBits)
     {
          mVNbB.push_back(ElMin(cFlagTabule<Type>::TheNbBits,aNbFlag-aK));
          mVSz.push_back(1<<mVNbB.back());
          mVIm1D.push_back(Im1D_REAL4(mVSz.back()));
          mDIm.push_back(mVIm1D.back().data());
     }
     mData = & (mDIm[0]);
}



template  <class Type> cFlagPonder<Type> * cFlagPonder<Type>::PonderSomFlag(const Pt2di & aV,double Depond,double aGama)
{
   return PonderSomFlag(NormalizeSom1(PondVois(aV,Depond,aGama)));
}

template  <class Type> cFlagPonder<Type>  *cFlagPonder<Type>::PonderSomFlag(const std::vector<double> & aV,const Pt2di & aVois)
{
   cFlagPonder<Type> * aRes = new cFlagPonder<Type>((int)aV.size(), aVois);
   int aNbBCum = 0;

   for (int aK=0 ; aK<int(aRes->mDIm.size()) ; aK++)
   {
        for (int aFlag=0 ; aFlag<aRes->mVSz[aK] ; aFlag++)
        {
             double aSom = 0;
             for (int aB=0 ; aB<aRes->mVNbB[aK] ; aB++)
             {
                 if (! (aFlag & (1<<aB)))
                 {
                    aSom += aV[aNbBCum+aB];
                 }
             }
             aRes->mData[aK][aFlag] = aSom;
        }

        aNbBCum += aRes->mVNbB[aK];
   }

   return aRes;
}

      /*********************************************************************/
      /*                                                                   */
      /*                     cImFlags<Type>                                */
      /*                                                                   */
      /*********************************************************************/


template   <class Type> cImFlags<Type>::cImFlags(Pt2di aSz,int aNbFlag) :
     cFlagTabule<Type> (aNbFlag),
     mSzIm    (aSz),
     mTxF     (aSz.x * cFlagTabule<Type>::mNbEl),
     mSzFlag  (mTxF,mSzIm.y),
     mIm      (mSzFlag.x,mSzFlag.y,tBase(0)),
     mData    (mIm.data())
{
}




template   <class Type> cImFlags<Type> cImFlags<Type>::CensusMS
                                  (
                                       const std::vector<Im2D_REAL4> & aVIm,
                                       Pt2di aSzVMax,
                                       const std::vector<Pt2di> &aVSz,
                                       const std::vector<double> &aVSigma
                                  )
{
    ELISE_ASSERT(aVIm.size()==aVSz.size(),"cImFlags<Type>::CensusMS size pb");
    ELISE_ASSERT(aVIm.size()==aVSigma.size(),"cImFlags<Type>::CensusMS size pb");

    if (aVIm.size() ==1)
       return cImFlags<Type>::Census(aVIm[0],aVSz[0]);

    Pt2di aSz = aVIm[0].sz();
    cImFlags<Type> aRes(aSz,NbSomOfVois(aSzVMax));
    int aNbScale = (int)aVSz.size();

    std::vector<int> aVKIm = VecKIm(aSzVMax,aVSz,aVSigma);

/*
    for (int anY=-aSzVMax.y ; anY<=aSzVMax.y ; anY++)
    {
        for (int anX=-aSzVMax.x ; anX<=aSzVMax.x ; anX++)
        {
            double aScaleMin = 1e6;
            int aKSMin = -1;
            for (int aKS=0 ; aKS<aNbScale ; aKS++)
            {
                Pt2di aSzV = aVSz[aKS];

                if ( (ElAbs(anX)<=aSzV.x) &&  (ElAbs(anY)<=aSzV.y) )
                {
                    double aScale = aVSigma[aKS];
                    if (aScale < aScaleMin)
                    {
                        aScaleMin = aScale;
                        aKSMin = aKS;
                    }
                }
            }
            ELISE_ASSERT(aKSMin>=0,"CensusMS no KS");
            aVKIm.push_back(aKSMin);

        }
    }
*/



    std::vector<cBufOnImage<float> * > aVBOI ;
    for (int aKS=0 ; aKS<aNbScale ; aKS++)
    {
        aVBOI.push_back(cBufOnImage<float>::FullBufOnIm(aVIm[aKS]));
    }

    for (int aXGlob=0 ; aXGlob < aSz.x - aSzVMax.x ; aXGlob++)
    {
         if (aXGlob>=aSzVMax.x)
         {
             for (int aYGlob=0 ; aYGlob < aSz.y - aSzVMax.y ; aYGlob++)
             {
                 if (aYGlob>=aSzVMax.y)
                 {
                      aRes.Init(Pt2di(aXGlob,aYGlob));
                      std::vector<float **> aVData;
                      for (int aKS=0 ; aKS<aNbScale ; aKS++)
                          aVData.push_back(aVBOI[aKS]->data());



                      float *** aData = &(aVData[0]);
                      float aV0 = aData[0][0][0];

                      int * aKIm = &(aVKIm[0]);
                      for (int anY=-aSzVMax.y ; anY<=aSzVMax.y ; anY++)
                      {
                          for (int anX=-aSzVMax.x ; anX<=aSzVMax.x ; anX++)
                          {
                               if (aV0<=aData[*aKIm][anY][anX])
                               {
                                  aRes.AddCurFlag();
                               }
                               aRes.Next();
                               aKIm++;
                          }
                      }
/*
*/
                 }
                 for (int aKS=0 ; aKS<aNbScale ; aKS++)
                     aVBOI[aKS]->AvanceY();
             }
         }
         for (int aKS=0 ; aKS<aNbScale ; aKS++)
             aVBOI[aKS]->AvanceX();
    }
    DeleteAndClear(aVBOI);

    return aRes;
}


template   <class Type> cImFlags<Type> cImFlags<Type>::Census(Im2D_REAL4 anIm,Pt2di aSzV)
{
    cImFlags<Type> aRes(anIm.sz(),NbSomOfVois(aSzV));
    Pt2di aSz = anIm.sz();
    cBufOnImage<float> * aBOI = cBufOnImage<float>::FullBufOnIm(anIm);
    // INT ** aDFlag = aRes.data();

    for (int anX=0 ; anX < aSz.x - aSzV.x ; anX++)
    {
         if (anX>=aSzV.x)
         {
             for (int anY=0 ; anY < aSz.y - aSzV.y ; anY++)
             {
                 if (anY>=aSzV.y)
                 {
                      aRes.Init(Pt2di(anX,anY));
                      float ** aData = aBOI->data();
                      float aV0 = aData[0][0];

if (0)
{
std::cout << "XXX " << aV0 << " " << anIm.data()[anY][anX] << "\n";
ELISE_ASSERT(aV0== anIm.data()[anY][anX], "JJJJj\n");
}
                      for (int anY=-aSzV.y ; anY<=aSzV.y ; anY++)
                      {
                          float * aL = aData[anY];
                          for (int anX=-aSzV.x ; anX<=aSzV.x ; anX++)
                          {
                               if (aV0<=aL[anX])
                               {
                                  aRes.AddCurFlag();
                               }
                               aRes.Next();
                          }
                      }
                 }
                 aBOI->AvanceY();
             }
         }
         aBOI->AvanceX();
    }
    delete aBOI;
   return aRes;
}




// static cImFlags<Type>  Census(Im2D_REAL4 anIm,Pt2di aSzV);


//cImFlags<U_INT2> aIIII(Pt2di(3,3),1);
//cImFlags<INT4>   aIIIIIIII(Pt2di(3,3),1);
/*
void fff()
{
   //cFlagPonder<U_INT2> aF(44);
   std::vector<double> aV;
   cFlagPonder<U_INT2> * aF = cFlagPonder<U_INT2>::PonderSomFlag(aV);
   delete aF;
}
*/


      /*********************************************************************/
      /*                                                                   */
      /*                   cBufOnImage<Type>                               */
      /*                                                                   */
      /*********************************************************************/


template <class Type> cBufOnImage<Type>::cBufOnImage(Type ** aDataIm,Box2di aBoxDef,Box2di aBoxCalc) :
   mBoxDef  (aBoxDef),
   mBoxCalc (aBoxCalc),
   mBoxTot  (Sup(mBoxDef,mBoxCalc))  
{
    
    mNbL = 0;

    for (int anY=mBoxTot._p0.y ; anY<mBoxTot._p1.y ; anY++)
    {
         if ( (anY>=mBoxDef._p0.y) && (anY<aBoxDef._p1.y))
             mVData.push_back(aDataIm[anY]+aBoxCalc._p0.x);
         else 
             mVData.push_back(0);

         mNbL++;
    }
    mData0 = &mVData[0];
    InitY();

}
template <class Type> void cBufOnImage<Type>::InitY()
{
    mData = mData0 -mBoxDef._p0.y + mBoxCalc._p0.y;
}

template <class Type> void cBufOnImage<Type>::AvanceX()
{
    InitY();
    for (int aK=0 ; aK<mNbL ; aK++)
        mData0[aK]++;
}
template <class Type>  cBufOnImage<Type> * cBufOnImage<Type>::FullBufOnIm(Im2D<Type,tBase> anIm)
{
     Box2di aBoxDef(Pt2di(0,0),anIm.sz());
     return new cBufOnImage<Type>(anIm.data(),aBoxDef,aBoxDef);
}


      /*********************************************************************/
      /*                                                                   */
      /*                   cCensusGr                                       */
      /*                                                                   */
      /*********************************************************************/


cCensusGr::cCensusGr(const Pt2di & aSzV,const double & FactDepond,bool DoFlag,cCensusGr * aGrShareFlag) :
   mFactDepond (FactDepond),
   mSomP  (0),
   mSzV   (aSzV),
   mNbSom (0),
   mCostFlag (1)
{
     for (int aDy1=-1 ; aDy1<=1 ; aDy1++)
     {
          for (int aDx1=-1 ; aDx1<= 1 ; aDx1++)
          {
              Pt2di aP1(aDx1*aSzV.x,aDy1*aSzV.y);
              cOnePCGr aPcGR(aP1);
              for (int aK=0 ; aK<4 ; aK++)
              {
                   int aDx2 = aP1.x+VX[aK]*aSzV.x;
                   int aDy2 = aP1.y+VY[aK]*aSzV.y;
                   if  ((aDx2>=-aSzV.x) && (aDx2<=aSzV.x) && (aDy2>=-aSzV.y) && (aDy2<=aSzV.y))
                   {
                       Pt2di aP2(aDx2,aDy2);

                       double aD = (euclid(aP1)+euclid(aP2)) / 2.0;
                       double aRatio = aD / euclid(aSzV);
                       double aPds = 1 - aRatio;
                       aPds = (1-FactDepond) + FactDepond * aPds;
                       int IPds = round_ni(aPds*mMulPds);

                       aPcGR.AddV(aP2,IPds,mNbSom);
                       mSomP += IPds;
                       mNbSom++;
                   }
              }
              mPCGR.push_back(aPcGR);

          }
     }
     ElTimer aChrono;
     if (DoFlag)
     {
         mNbFlag = 1 << mNbSom;
         if (aGrShareFlag)
         {
             ELISE_ASSERT(aGrShareFlag->mFactDepond==mFactDepond,"Fact Depond duf in Gr Share Flag");
             mCostFlag = aGrShareFlag->mCostFlag;
             mDCF      = aGrShareFlag->mDCF;
         }
         else
         {
             mCostFlag.Resize(mNbFlag);
             mDCF =  mCostFlag.data();
             for (int aFlag=0 ; aFlag<mNbFlag; aFlag++)
             {
                 int aSomP = 0;
                 for (int aK1=0 ; aK1<int(mPCGR.size()) ; aK1++)
                 {
                    cOnePCGr & aSG = mPCGR[aK1];
                    for (int aK2=0 ; aK2<int(aSG.mV.size()) ; aK2++)
                    {
                         if (!(aFlag & (aSG.mV[aK2].mFlag)))
                         {
                             aSomP += aSG.mV[aK2].mP;
                         }
                    }
                 }
                 mDCF[aFlag] = aSomP;
//std::cout << "FLGAP " << aFlag << " " << aSomP << "\n";
             }
         }
     }
//getchar();
}

Im2D_INT4 cCensusGr::CalcFlag(Im2D_REAL4 anIm)
{
    Pt2di aSz = anIm.sz();
    Im2D_INT4 aRes(aSz.x,aSz.y);
    cBufOnImage<float> * aBOI = cBufOnImage<float>::FullBufOnIm(anIm);
    INT ** aDFlag = aRes.data();

    for (int anX=0 ; anX < aSz.x - mSzV.x ; anX++)
    {
         if (anX>=mSzV.x)
         {
             for (int anY=0 ; anY < aSz.y - mSzV.y ; anY++)
             {
                 if (anY>=mSzV.y)
                 {
                     aDFlag[anY][anX] = FlagVois(aBOI->data());
                 }
                 aBOI->AvanceY();
             }
         }
         aBOI->AvanceX();
    }
    delete aBOI;
   return aRes;
}

int cCensusGr::FlagVois(float ** Im1)
{
   int aFlag=0;
   for (int aK1=0 ; aK1<int(mPCGR.size()) ; aK1++)
   {
       cOnePCGr & aSG = mPCGR[aK1];
       const Pt2di & aP1 = aSG.mP0;
       float aV1 = Im1[aP1.y][aP1.x];
       for (int aK2=0 ; aK2<int(aSG.mV.size()) ; aK2++)
       {
           const Pt2di & aP2 = aSG.mV[aK2].mV; 
           float aW1 = Im1[aP2.y][aP2.x];

           if (aV1<aW1) 
              aFlag  |=  aSG.mV[aK2].mFlag;
       }
   }


   return aFlag;
}


double cCensusGr::GainBasic(float ** Im1,float ** Im2,int  aPx2)
{
   int aSomLoc=0;
   for (int aK1=0 ; aK1<int(mPCGR.size()) ; aK1++)
   {
       cOnePCGr & aSG = mPCGR[aK1];
       const Pt2di & aP1 = aSG.mP0;
       float aV1 = Im1[aP1.y][aP1.x];
       float aV2 = Im2[aP1.y][aP1.x+aPx2];
       for (int aK2=0 ; aK2<int(aSG.mV.size()) ; aK2++)
       {
           const Pt2di & aP2 = aSG.mV[aK2].mV; 
           float aW1 = Im1[aP2.y][aP2.x];
           float aW2 = Im2[aP2.y][aP2.x+aPx2];

           if ((aV1<aW1) == (aV2<aW2))
              aSomLoc+= aSG.mV[aK2].mP;
       }
   }


   return aSomLoc / double(mSomP);
}

      /*********************************************************************/
      /*                                                                   */
      /*                      cMoment_Correl                               */
      /*                                                                   */
      /*********************************************************************/

typedef double  tMomC;

class cMoment_Correl
{
    public :
           cMoment_Correl(
             const std::vector<Im2D_REAL4>          & aVIm,
             const std::vector<std::vector<Pt2di> > & aVV,
             const std::vector<double > &             aVPds,
             Pt2di aSzMax
           );
           tMomC *** DataSom() {return mData1;}
           tMomC *** DataSomQuad() {return mData2;}
    private :
         cMoment_Correl(const cMoment_Correl &); // N.I.
         int mNbIm;
         std::vector<Im2D<tMomC,double> > mVS1;
         std::vector<tMomC **>  mVData1;
         tMomC ***              mData1;
         std::vector<Im2D<tMomC,double> > mVS2;
         std::vector<tMomC **>  mVData2;
         tMomC ***              mData2;
};

cMoment_Correl::cMoment_Correl
(
             const std::vector<Im2D_REAL4>          & aVIm,
             const std::vector<std::vector<Pt2di> > & aVV,
             const std::vector<double > &             aVPds,
             Pt2di                                    aSzVMax
) :
   mNbIm ((int)aVIm.size())
{
    Pt2di aSz = aVIm[0].sz();
    std::vector<cBufOnImage<float> * > aVBOI ;
    for (int aKS=0 ; aKS<mNbIm ; aKS++)
    {
        aVBOI.push_back(cBufOnImage<float>::FullBufOnIm(aVIm[aKS]));
        mVS1.push_back(Im2D<tMomC,double>(aSz.x,aSz.y));
        mVData1.push_back(mVS1.back().data());
        mVS2.push_back(Im2D<tMomC,double>(aSz.x,aSz.y));
        mVData2.push_back(mVS2.back().data());
    }
    mData1 = &(mVData1[0]);
    mData2 = &(mVData2[0]);
   

    for (int aXGlob=0 ; aXGlob < aSz.x - aSzVMax.x ; aXGlob++)
    {
         if (aXGlob>=aSzVMax.x)
         {
             for (int aYGlob=0 ; aYGlob < aSz.y - aSzVMax.y ; aYGlob++)
             {
                 if (aYGlob>=aSzVMax.y)
                 {
                      double aGlobSom1 = 0;
                      double aGlobSom2 = 0;
                      double aGlobPds = 0;
                      for (int aKS=0 ; aKS<mNbIm ; aKS++)
                      {
                          double aSom1 = 0;
                          double aSom2 = 0;

                          const std::vector<Pt2di> & aVP = aVV[aKS];
                          int aNbP = (int)aVP.size();
                          float ** anIm = aVBOI[aKS]->data();
                          double aPdsK = aVPds[aKS];
                          for (int aKP=0 ; aKP<aNbP ; aKP++)
                          {
                              const Pt2di aP = aVP[aKP];
                              float aV = anIm[aP.y][aP.x];
                              aSom1 += aV;
                              aSom2 += ElSquare(aV);
                          }

                          aGlobSom1 += aSom1 * aPdsK;
                          aGlobSom2 += aSom2 * aPdsK;
                          aGlobPds += aPdsK * aNbP;
                          mData1[aKS][aYGlob][aXGlob] = aGlobSom1 / aGlobPds;
                          mData2[aKS][aYGlob][aXGlob] = aGlobSom2 / aGlobPds;
                      }
                      
                 }
                 for (int aKS=0 ; aKS<mNbIm ; aKS++)
                     aVBOI[aKS]->AvanceY();
             }
         }
         for (int aKS=0 ; aKS<mNbIm ; aKS++)
             aVBOI[aKS]->AvanceX();
    }
    DeleteAndClear(aVBOI);

}



      /*********************************************************************/
      /*                                                                   */
      /*                      ::                                           */
      /*                                                                   */
      /*********************************************************************/


double CorrelBasic(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV,float anEpsilon)
{
     cQckInterpolEpip aQI2(X2);
     RMat_Inertie aMat;
     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
               aMat.add_pt_en_place(aL1[aDx],aQI2.GetVal(aL2+aDx));
          }
     }
     return aMat.correlation(anEpsilon);
}

double CorrelBasic_ImInt(float ** Im1,Pt2di aP1,float ** Im2,Pt2di aP2,Pt2di aSzV,float anEpsilon)
{
     RMat_Inertie aMat;
     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[aP2.y+aDy] + aP2.x;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
               aMat.add_pt_en_place(aL1[aDx],aL2[aDx]);
          }
     }
     return aMat.correlation(anEpsilon);
}

double CorrelBasic_Center(float ** Im1,float ** Im2,int  aPx2,Pt2di aSzV,float anEpsilon)
{
     RMat_Inertie aMat;
     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aDy] ;
          float * aL2 = Im2[aDy] + aPx2;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
               aMat.add_pt_en_place(aL1[aDx],aL2[aDx]);
          }
     }
     return aMat.correlation(anEpsilon);
}

double MS_CorrelBasic_Center
       (
             const std::vector<cBufOnImage<float> *> & aVBOI1,
             const std::vector<cBufOnImage<float> *> & aVBOI2,
             int  aPx2,
             const std::vector<std::vector<Pt2di> > & aVV,
             const std::vector<double > &             aVPds,
             double anEpsilon,
             bool  ModeMax
       )
{
     RMat_Inertie aMat;
     double aMaxCor = -1;
     for (int aKS=0 ; aKS< int(aVV.size()) ; aKS++)
     {
          const std::vector<Pt2di> & aVP = aVV[aKS];
          double aPds = aVPds[aKS];
          int aNbP = (int)aVP.size();
          float ** anIm1 = aVBOI1[aKS]->data();
          float ** anIm2 = aVBOI2[aKS]->data();
          for (int aKP=0 ; aKP<aNbP ; aKP++)
          {
              const Pt2di aP = aVP[aKP];
              aMat.add_pt_en_place(anIm1[aP.y][aP.x],anIm2[aP.y][aP.x+aPx2],aPds);
          }
          if (ModeMax) 
          {
             ElSetMax(aMaxCor,aMat.correlation(anEpsilon));
          }
     }
     if (ModeMax) 
        return aMaxCor;
     return aMat.correlation(anEpsilon);
}

double Quick_MS_CensusQuant
       (
             const std::vector<cBufOnImage<float> *> & aVBOI1,
             const std::vector<cBufOnImage<float> *> & aVBOI2,
             int  aPx2,
             const std::vector<std::vector<Pt2di> > & aVV,
             const std::vector<double > &             aVPds
       )
{
   
        //      aSomEcart += ElAbs(EcartNormalise(aV1,aVC1)-EcartNormalise(aV2,aVC2));
     float aVC1 = aVBOI1[0]->data()[0][0];
     float aVC2 = aVBOI2[0]->data()[0][aPx2];
     double aSomEcGlob = 0;
     double aPdsGlob = 0;
     int aNbScale = (int)aVV.size();
     for (int aKS=0 ; aKS< aNbScale ; aKS++)
     {
          const std::vector<Pt2di> & aVP = aVV[aKS];
          double aPds = aVPds[aKS];
          double aSomEc = 0;
          int aNbP = (int)aVP.size();
          float ** anIm1 = aVBOI1[aKS]->data();
          float ** anIm2 = aVBOI2[aKS]->data();
          for (int aKP=0 ; aKP<aNbP ; aKP++)
          {
              const Pt2di aP = aVP[aKP];
              aSomEc += ElAbs(EcartNormalise(aVC1,anIm1[aP.y][aP.x])-EcartNormalise(aVC2,anIm2[aP.y][aP.x+aPx2]));
          }
          aSomEcGlob += aSomEc * aPds;
          aPdsGlob += aPds * aNbP;
     }
     double anEc = aSomEcGlob /aPdsGlob;
     // anEc = ElMin100;
     return 1 - anEc;
}


double Quick_MS_CorrelBasic_Center
       (
             const Pt2di & aPG1,
             const Pt2di & aPG2,
             double ***  aSom1,
             double ***  aSom11,
             double ***  aSom2,
             double ***  aSom22,
             const std::vector<cBufOnImage<float> *> & aVBOI1,
             const std::vector<cBufOnImage<float> *> & aVBOI2,
             int  aPx2,
             const std::vector<std::vector<Pt2di> > & aVV,
             const std::vector<double > &             aVPds,
             double anEpsilon,
             bool  ModeMax
       )
{

     if (MPD_MM())
     {
         // std::cout << " MODE-MAX=" << ModeMax << " SZs " << aVBOI1.size() << aVBOI2.size() << aVV.size() << aVPds.size() << "\n";
         // MODE-MAX=1 SZs 3333
/*
         std::cout << aVPds << " " << aVV << "\n";
         getchar();
[0.111111,0.02,0.00396825] 
[
  [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],
  [[-2,-2],[-1,-2],[0,-2],[1,-2],[2,-2],[-2,-1],[2,-1],[-2,0],[2,0],[-2,1],[2,1],[-2,2],[-1,2],[0,2],[1,2],[2,2]],
  [[-4,-3],[-3,-3],[-2,-3],[-1,-3],[0,-3],[1,-3],[2,-3],[3,-3],[4,-3],[-4,-2],[-3,-2],[3,-2],[4,-2],[-4,-1],[-3,-1],[3,-1],[4,-1],[-4,0],[-3,0],[3,0],[4,0],[-4,1],[-3,1],[3,1],[4,1],[-4,2],[-3,2],[3,2],[4,2],[-4,3],[-3,3],[-2,3],[-1,3],[0,3],[1,3],[2,3],[3,3],[4,3]]]

*/
     }

     double aMaxCor = -1;
     double aCovGlob = 0;
     double aPdsGlob = 0;
     int aNbScale = (int)aVV.size();
     for (int aKS=0 ; aKS< aNbScale ; aKS++)
     {
          bool aLast = (aKS==(aNbScale-1));
          const std::vector<Pt2di> & aVP = aVV[aKS];
          double aPds = aVPds[aKS];
          double aCov = 0;
          int aNbP = (int)aVP.size();
          float ** anIm1 = aVBOI1[aKS]->data();
          float ** anIm2 = aVBOI2[aKS]->data();
          aPdsGlob += aPds * aNbP;
          for (int aKP=0 ; aKP<aNbP ; aKP++)
          {
              const Pt2di aP = aVP[aKP];
              aCov += anIm1[aP.y][aP.x]*anIm2[aP.y][aP.x+aPx2];

          }
          aCovGlob += aCov * aPds;

          if (ModeMax || aLast)
          {
              double aM1 = aSom1[aKS][aPG1.y][aPG1.x];
              double aM2 = aSom2[aKS][aPG2.y][aPG2.x];
              double aM11 = aSom11[aKS][aPG1.y][aPG1.x] - ElSquare(aM1);
              double aM22 = aSom22[aKS][aPG2.y][aPG2.x] - ElSquare(aM2);
              double aM12 = aCovGlob / aPdsGlob - aM1 * aM2;

              if (ModeMax) 
              {
                 double aCor = (aM12 * ElAbs(aM12)) /ElMax(anEpsilon,aM11*aM22);
                 ElSetMax(aMaxCor,aCor);
              }
              else
                 return aM12 / sqrt(ElMax(anEpsilon,aM11*aM22));
         }

          
     }
     return (aMaxCor > 0) ? sqrt(aMaxCor) : - sqrt(-aMaxCor) ;
}







double CensusBasicCenter(float ** Im1,float ** Im2,int aPx2,Pt2di aSzV)
{
     float aC1 = **Im1;
     float aC2 = Im2[0][aPx2];
     int aNbOk = 0;
     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aDy] ;
          float * aL2 = Im2[aDy] + aPx2;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              bool Inf1 = (aL1[aDx]<aC1);
              bool Inf2 = (aL2[aDx]<aC2);
              if (Inf1==Inf2) aNbOk++;
          }
     }
     return ((double) aNbOk) / ((1+2*aSzV.x)*(1+2*aSzV.y));
}

double CensusBasic(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV)
{
     cQckInterpolEpip aQI2(X2);
     float aC1 =  Im1[aP1.y][aP1.x];
     float aC2 =  aQI2.GetVal(Im2[aP1.y]+ aQI2.mX0);
     int aNbOk = 0;


     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              float aV1 = aL1[aDx];
              float aV2 = aQI2.GetVal(aL2+aDx);
              // if ((aDx==0) && (aDy==0)) std::cout << "TTTt " << (aC1-aV1) << " " << (aC2-aV2) << "\n";

              bool Inf1 = (aV1<aC1);
              bool Inf2 = (aV2<aC2);
              if (Inf1==Inf2) aNbOk++;
          }
     }
     return ((double) aNbOk) / ((1+2*aSzV.x)*(1+2*aSzV.y));
}

double CensusQuantif(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV)
{
     cQckInterpolEpip aQI2(X2);

     float aVC1 =  Im1[aP1.y][aP1.x];
     float * aL2C = Im2[aP1.y] + aQI2.mX0; // debut de la colone Im2, centre sur la partie entiere
     float aVC2 = aQI2.GetVal(aL2C);  // 
     double aSomEcart = 0;

     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x; // debut de la colone Im1, centre x1
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0; // debut de la colone Im2, centre sur la partie entiere
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              // Val du voisin  Dx,Dy en 
              float aV1 = aL1[aDx]; // 
              float aV2 = aQI2.GetVal(aL2+aDx);  // 

              aSomEcart += ElAbs(EcartNormalise(aV1,aVC1)-EcartNormalise(aV2,aVC2));
          }
     }
     return aSomEcart / ( (1+2*aSzV.x) * (1+2*aSzV.y) );
}


Im2D_REAL4 AutoCorr_CensusQuant(Im2D_REAL4 anImIn,Pt2di aSzW,double aDPx)
{
   Pt2di aSzIm = anImIn.sz();
   Im2D_REAL4 aRes(aSzIm.x,aSzIm.y,0.0);

   for (int aX=aSzW.x ; aX<aSzIm.x -aSzW.x ; aX++)
   {
      for (int aY=aSzW.y+1 ; aY<aSzIm.y-aSzW.y-1 ; aY++)
      {
          double aC1 = CensusQuantif(anImIn.data(),Pt2di(aX,aY),anImIn.data(),aX+aDPx,aY,aSzW);
          double aC2 = CensusQuantif(anImIn.data(),Pt2di(aX,aY),anImIn.data(),aX+aDPx,aY,aSzW);
          double aC = (aC1+aC2) / 2.0;
          // aC = 1-aC;
          aC /= aDPx;
          aRes.SetR(Pt2di(aX,aY),aC);
      }
   }
   return aRes;
}

int CPP_AutoCorr_CensusQuant(int argc,char ** argv)
{
    std::string aNameIn,aNameOut;
    int aSzW=2;
    double aEps=0.1;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aNameIn,"Name of Input image", eSAM_IsPatFile),
           LArgMain() << EAM(aSzW,"SzW",true,"Size of Window, def=2")
                      << EAM(aEps,"Eps",true,"Size of epsilon")
                      << EAM(aNameOut,"Out",true,"Name of output")

    );

    if (! EAMIsInit(&aNameOut))
       aNameOut = "AC-CensusQ-" + StdPrefix(aNameIn) + ".tif";

    Im2D_REAL4 aImIn = Im2D_REAL4::FromFileStd(aNameIn);
    Im2D_REAL4 aRes = AutoCorr_CensusQuant(aImIn,Pt2di(aSzW,aSzW),aEps);

    Tiff_Im::CreateFromIm(aRes,aNameOut);

    return EXIT_SUCCESS;
}


/*
*/


// Version basique du calcul de Census par graphe;
// Est utilise pour verifier la correction du calcul optimise


double CensusGraphePlein(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV)
{
     cQckInterpolEpip aQI2(X2);
     int aNbOk = 0;
     int aNbMiss = 0;


     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x; // debut de la colone Im1, centre x1
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0; // debut de la colone Im2, centre sur la partie entiere
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              // Val du voisin  Dx,Dy en 
              float aV1 = aL1[aDx]; // 
              float aV2 = aQI2.GetVal(aL2+aDx);  // 
              for (int aK=0 ; aK<4 ; aK++)
              {
                   int aDx2 = aDx+VX[aK];  // Dx-Dy des des 8 voisins
                   int aDy2 = aDy+VY[aK];
                   // Pour ne pas sortir
                   if  ((aDx2>=-aSzV.x) && (aDx2<=aSzV.x) && (aDy2>=-aSzV.y) && (aDy2<=aSzV.y))
                   {
                       // 
                       float aW1 = Im1[aP1.y+aDy2][aP1.x+aDx2];
                       float aW2 = aQI2.GetVal(Im2,Pt2di(aDx2,Y2+aDy2));
                       bool Inf1 = (aV1<aW1);
                       bool Inf2 = (aV2<aW2);
                       
                       if (Inf1==Inf2)
                          aNbOk++;
                       else          
                           aNbMiss++;
                   }
              }

          }
     }
     return ((double) aNbOk) / (aNbOk+aNbMiss);
}

double CensusGraphe_ImInt(float ** Im1,Pt2di aP1,float ** Im2,Pt2di aP2,Pt2di aSzV)
{
     int aNbOk = 0;
     int aNbMiss = 0;


     for (int aDyA=-aSzV.y ; aDyA<=aSzV.y ; aDyA += aSzV.y)
     {
          float * aL1 = Im1[aP1.y+aDyA] + aP1.x;
          float * aL2 = Im2[aP2.y+aDyA] + aP2.x;
          for (int aDxA=-aSzV.x ; aDxA<= aSzV.x ; aDxA+=aSzV.x)
          {
              float aV1 = aL1[aDxA];
              float aV2 = aL2[aDxA];
              for (int aK=0 ; aK<4 ; aK++)
              {
                   int aDxB = aDxA+VX[aK]*aSzV.x;
                   int aDyB = aDyA+VY[aK]*aSzV.y;
                   if  ((aDxB>=-aSzV.x) && (aDxB<=aSzV.x) && (aDyB>=-aSzV.y) && (aDyB<=aSzV.y))
                   {
                       float aW1 = Im1[aP1.y+aDyB][aP1.x+aDxB];
                       float aW2 = Im2[aP2.y+aDyB][aP2.x+aDxB];
                       bool Inf1 = (aV1<aW1);
                       bool Inf2 = (aV2<aW2);
                       
                       if (Inf1==Inf2)
                          aNbOk++;
                       else          
                           aNbMiss++;
                   }
              }

          }
     }
     return ((double) aNbOk) / (aNbOk+aNbMiss);
}






     // float aValStd = aQI2.GetVal(aDataIm1,Pt2di(0,anY+anOff1.y));
     // float aValNew = aDataC[anY+anOff1.y][anX+anOff1.x+anOffset];

      /*********************************************************************/
      /*                                                                   */
      /*                      cAppliMICMAC                                 */
      /*                                                                   */
      /*********************************************************************/



double TolNbByPix=1e-5;
void cAppliMICMAC::DoCensusCorrel(const Box2di & aBox,const cCensusCost & aCC)
{

   
  bool Verif = aCC.Verif().Val();
  bool DoMixte =  (aCC.TypeCost() == eMCC_CensusMixCorrelBasic);
  bool DoGraphe = (aCC.TypeCost() ==eMCC_GrCensus);
  bool DoCensusBasic = (aCC.TypeCost() ==eMCC_CensusBasic) || DoMixte;
  bool DoCorrel = (aCC.TypeCost() == eMCC_CensusCorrel) || DoMixte;
  bool DoCensQuant = (aCC.TypeCost() == eMCC_CensusQuantitatif );

  double aDynCensusCost = aCC.Dyn().Val();


  // return Quick_MS_CensusQuant(aVBOI1,aVBOI2,aPx2,aVV,aVPds);

  if (MPD_MM())
  {
      //  std::cout << "HHHHHHHHHHHHHHHHHHHh " << aCC.TypeCost() << " "<<  eMCC_CensusCorrel << " VERIF " << Verif << "\n";
      // getchar(); // => HHHHHHHHHHHHHHHHHHHh 2 2 VERIF 0
  }

  double aSeuilHC = aCC.SeuilHautCorMixte().Val();
  double aSeuilBC = aCC.SeuilBasCorMixte().Val();

  bool aModeMax = false;

   std::vector<float> aVPmsInit;
   double aSomPmsInit=0;

   std::vector<Pt2di>     aVSz;
   std::vector<double>    aVSigma;
   std::vector<double>    aVPsdInit;
   if (CMS())
   {
       const std::vector<cOneParamCMS> & aVP = CMS()->OneParamCMS();
       for (int aK=0 ; aK<int(aVP.size()) ; aK++)
       {
          aVPmsInit.push_back(aVP[aK].Pds());
          aSomPmsInit += aVPmsInit.back();
          aVSigma.push_back(aVP[aK].Sigma());
          aVSz.push_back(aVP[aK].SzW());
          // std::cout << "SOMP " << aSomPms << "\n";
       }
       aModeMax = CMS()->ModeMax().Val();
   }
   else
   {
       aVPmsInit.push_back(1.0);
       aSomPmsInit = 1.0;
       aVSigma.push_back(0.0);
       aVSz.push_back(mCurSzV0);
   }
   std::vector<std::vector<Pt2di> > aVKImS = VecKImSplit(mCurSzVMax,aVSz,aVSigma);
   std::vector<double> aVPds;
   for (int aKS=0 ; aKS<int(aVSz.size()) ; aKS++)
   {
      aVPds.push_back(aVPmsInit[aKS]/NbSomOfVois(aVSz[aKS]));
   }

   int aNbScale = (int)aVPmsInit.size();
   float * aDataPmsInit = & (aVPmsInit[0]);


   cGPU_LoadedImGeom &   anI0 = *(mVLI[0]);
   cGPU_LoadedImGeom &   anI1 = *(mVLI[1]);
   const std::vector<cGPU_LoadedImGeom *> & aVSLGI0 = anI0.MSGLI();

   ELISE_ASSERT((mX0Ter==0)&&(mY0Ter==0),"Origin Assumption in cAppliMICMAC::DoCensusCorrel");
   // Pt2di aSzT = aBox.sz();

  //  Censur Graphe
   bool DoFlag = DoGraphe;
   std::vector<cCensusGr *> aVCG;
   cCensusGr * aCG = 0;
   if (DoGraphe)
   {
       for (int aK=0 ; aK<int(aVSLGI0.size()) ; aK++)
       {
            aVCG.push_back(new cCensusGr(aVSLGI0[aK]->SzV0(),0.0,DoFlag, (aK==0)?0:aVCG[0]));
       }
       aCG = aVCG[0];
   }

   cMoment_Correl * aMom1 = 0;
   tMomC ***        aSom1 = 0;
   tMomC ***        aSom11 = 0;
   if (DoCorrel)
   {
       aMom1  = new cMoment_Correl(anI0.VIm(),aVKImS,aVPds,mCurSzVMax);
       aSom1  = aMom1->DataSom();
       aSom11 = aMom1->DataSomQuad();
   }


 //  ====  VERIFICATION DYNAMIQUE DES ARGUMENTS ==========

 //  ====  1. GEOMETRIE EPIPOLAIRE BASIQUE
    ELISE_ASSERT
    (
         GeomImages() == eGeomImage_EpipolairePure,
         "Not epipolar geometry for census "
    );


 //  ====  1. GEOMETRIE EPIPOLAIRE BASIQUE
    ELISE_ASSERT
    (
       mNbIm <= 2,
       "Image > 2  in Census"
    );



    double aStepPix = mStepZ / mCurEtape->DeZoomTer();



 //  ====  2. Pas quotient d'entier
    double aRealNbByPix = 1/ aStepPix;
    int mNbByPix = round_ni(aRealNbByPix);

    if (ElAbs(aRealNbByPix-mNbByPix) > TolNbByPix)
    {
         std::cout << "For Step = " << mStepZ  << " GotDif " << aRealNbByPix-mNbByPix << "\n";
         ELISE_ASSERT(false,"in DoCensusCorre step is not 1/INT");
    }
/*
*/

    Pt2di anOff0 = anI0.OffsetIm();
    Pt2di anOff1 = anI1.OffsetIm();

#ifdef CUDA_ENABLED

	bool useGpu = CMS()->UseGpGpu().Val();

	if(useGpu)
	{		

		bool dynRegulGpu = CurEtape()->AlgoRegul() == eAlgoTestGPU;

		cGBV2_ProgDynOptimiseur* gpuOpt		= dynRegulGpu ? (cGBV2_ProgDynOptimiseur*)mSurfOpt	: NULL;
		InterfOptimizGpGpu*		 IGpuOpt	= dynRegulGpu ? gpuOpt->getInterfaceGpGpu()			: NULL;


		interface_Census_GPU.transfertImageAndMask(
					toUi2(mPDV1->LoadedIm().SzIm()),
					toUi2(mPDV2->LoadedIm().SzIm()),
					anI0.VDataIm(),
					anI1.VDataIm(),
					anI0.ImMasqErod(),
					anI1.ImMasqErod());

		interface_Census_GPU.init(
					Rect(mX0Ter,mY0Ter,mX1Ter,mY1Ter),
					aVKImS,
					aVPds,
					toInt2(anOff0),
					toInt2(anOff1),
					toUi2(mPDV1->LoadedIm().SzIm()),
					toUi2(mPDV2->LoadedIm().SzIm()),
					mTabZMin,
					mTabZMax,
					mNbByPix,
					aStepPix,
					mAhEpsilon,
					mAhDefCost,
					aSeuilHC,
					aSeuilBC,
					aModeMax,
					DoMixte,
					dynRegulGpu,
					IGpuOpt
					);

		interface_Census_GPU.Job_Correlation_MultiScale();

		GpGpuTools::NvtxR_Push("Start copy cost",0xFFAAFF33);

		if(dynRegulGpu)
		{
			for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
				for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
				{
					int aZ0		=  mTabZMin[anY][anX];
					int aZ1		=  mTabZMax[anY][anX];
					int delTaZ	= abs(aZ0-aZ1);
					bool bIMinZ = delTaZ < 512;
					Pt2di aPIm0 = Pt2di(anX,anY) + anOff0;
					bool OkIm0	= anI0.IsOkErod(aPIm0.x,aPIm0.y);

					if(OkIm0 && bIMinZ)
					{
						uint2 pt		= make_uint2(anX- mX0Ter,anY- mY0Ter);
						ushort* aCost	= interface_Census_GPU.getCost<ushort>(pt);
						pixel*  pix		= interface_Census_GPU.getCost<pixel>(pt);

						gpuOpt->gLocal_SetCout(Pt2di(anX,anY),aCost,pix);
					}
				}
		}
		else
		{
			for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
				for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
				{
					int aZ0		=  mTabZMin[anY][anX];
					int aZ1		=  mTabZMax[anY][anX];
					int delTaZ	= abs(aZ0-aZ1);
					bool bIMinZ = delTaZ < 512;
					Pt2di aPIm0 = Pt2di(anX,anY) + anOff0;
					bool OkIm0	= anI0.IsOkErod(aPIm0.x,aPIm0.y);

					for (int aZI=aZ0 ; aZI< aZ1 ; aZI++)
					{
						if(bIMinZ)
						{
							uint3 pt =make_uint3(anX- mX0Ter,anY- mY0Ter,aZI-aZ0);
							double aCost = interface_Census_GPU.getCost<float>(pt);
							mSurfOpt->SetCout(Pt2di(anX,anY),&aZI, aCost >= 0.f/* &&aCost <= 2.f*/ &&  OkIm0 ? aCost : mAhDefCost);
						}
						else
						{
							mSurfOpt->SetCout(Pt2di(anX,anY),&aZI, mAhDefCost);
						}
					}
				}
		}

		GpGpuTools::Nvtx_RangePop();

		interface_Census_GPU.dealloc();

		return;
	}

	GpGpuTools::NvtxR_Push("CPU MSC",0xFFAA0033);

#endif
// std::cout << anOff0 << anOff1 << "\n";

    // std::cout << mX0Ter  << " " << mY0Ter << "\n";

    // mCurSzVMax 

    float ** aDataIm0 =  anI0.VDataIm()[0];
    float ** aDataIm1 =  anI1.VDataIm()[0];
    // cInterpolateurIm2D<float> * anInt = CurEtape()->InterpFloat();


    Box2di aBoxCalc0 = aBox.trans(anOff0);
    Box2di aBoxCalc1 = aBox.trans(anOff1);

    Box2di aBoxDef0 (Pt2di(0,0),mPDV1->LoadedIm().SzIm());
    Box2di aBoxDef1 (Pt2di(0,0),mPDV2->LoadedIm().SzIm());
    

    std::vector<Im2D_INT4> mImFlag0;
    std::vector<INT4 **  > mVIF0 ;
    INT4 ***               mDIF0=0;

    cImFlags<U_INT2>   aTabFlag0 (Pt2di(1,1),1);
    cFlagPonder<U_INT2>  *  aPondFlag = 0;
    if ( DoCensusBasic) 
    {
        aPondFlag =  cFlagPonder<U_INT2>::PonderSomFlag(mCurSzVMax,aCC.AttenDist().Val(),1.0);

        //  aTabFlag0 =   cImFlags<U_INT2>::Census(*(anI0.FloatIm(0)),mCurSzVMax) ;
        aTabFlag0 = cImFlags<U_INT2>::CensusMS(anI0.VIm(),mCurSzVMax,aVSz,aVSigma);
    }
    // La phase code le decalage sub pixel, on impose un pas en 1/N pour n'avoir que N image 
    // interpolee a creer
    for (int aPhase = 0 ; aPhase<mNbByPix ; aPhase++)
    {

        // {if (MPD_MM()) { std::cout << "Phhh " << aPhase << "\n"; getchar(); }}
        // Au depart
        //     toujours Ph0
        //     ensuite Ph0/ Ph1
        
        std::vector<Im2D_INT4> mImFlag1;
        std::vector<INT4 **  > mVIF1;
        INT4 ***               mDIF1=0;
        int aPhaseCompl = mNbByPix - aPhase;
      
        for (int aK=0 ; aK<int(mBufCensusIm2.size()) ; aK++)
        {
            float ** aDataIm1 = anI1.VDataIm()[aK]; 
            float ** aDataC   = mDataBufC[aK];
            Pt2di aSz = mBufCensusIm2[aK].sz();

            // On calcule l'image interpolee
            for (int anY = 0 ; anY < aSz.y ; anY++)
            {
                 float * aL1 = aDataIm1[anY] ;
                 float * aC1 = aDataC[anY] ;
                 if (aPhase!=0)
                 {
                    int aNbX = aSz.x-1;
                    for (int anX=0 ; anX<aNbX ; anX++)
                    {
                        *aC1 =  (aPhase * aL1[1] + aPhaseCompl*aL1[0]) / mNbByPix;
                        aL1++;
                        aC1++;
                    }
                }
                else
                {
                   memcpy(aC1,aL1,sizeof(*aC1)*aSz.x);
                }
            }
            if (DoFlag)
            {

                 mImFlag1.push_back(aCG->CalcFlag(mBufCensusIm2[aK]));
                 mVIF1.push_back( mImFlag1.back().data());
                 if (aPhase==0)
                 {
                    mImFlag0.push_back(aCG->CalcFlag(*anI0.FloatIm(aK)));
                    mVIF0.push_back( mImFlag0.back().data());
                 }
            }

            if (DoFlag)
            {

                 if (aPhase==0)
                 {
                 }
            }
             
        }
        if (DoFlag)
        {
            mDIF0 = &(mVIF0[0]);
            mDIF1 = &(mVIF1[0]);
        }
        // float ** aDataC =  mDataBufC[0];


        std::vector<cBufOnImage<float> *> aVBOI0;
        std::vector<cBufOnImage<float> *> aVBOIC;
        if (DoCensusBasic || DoCorrel || DoCensQuant)
        {
             for (int aKC=0 ; aKC<aNbScale ; aKC++)
             {
                 aVBOI0.push_back(new  cBufOnImage<float> (anI0.VDataIm()[aKC],aBoxDef0,aBoxCalc0));
                 aVBOIC.push_back(new  cBufOnImage<float> (     mDataBufC[aKC],aBoxDef1,aBoxCalc1));
             }
             // aTabFlag1 =   cImFlags<U_INT2>::Census(mBufCensusIm2[0],mCurSzVMax) ;
        }

        cImFlags<U_INT2>   aTabFlag1 (Pt2di(1,1),1);
        if (DoCensusBasic )
        {
             aTabFlag1 = cImFlags<U_INT2>::CensusMS(mBufCensusIm2,mCurSzVMax,aVSz,aVSigma);
        }
        int aNbBOI = (int)aVBOI0.size();

        // cBufOnImage<float> aBOI0(aDataIm0,aBoxDef0,aBoxCalc0);
        // cBufOnImage<float> aBOIC(aDataC  ,aBoxDef1,aBoxCalc1);
 
        cMoment_Correl * aMomC = 0;
        tMomC ***        aSomC = 0;
        tMomC ***        aSomCC = 0;
        if (DoCorrel)
        {
            aMomC = new cMoment_Correl(mBufCensusIm2,aVKImS,aVPds,mCurSzVMax);
            aSomC = aMomC->DataSom();
            aSomCC = aMomC->DataSomQuad();
        }

        for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
        {
            for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
            {
                Pt2di aPIm0 = Pt2di(anX,anY) + anOff0;
                bool OkIm0 = anI0.IsOkErod(aPIm0.x,aPIm0.y);
                int aZ0 =  mTabZMin[anY][anX];
                int aZ1 =  mTabZMax[anY][anX];

                int aXIm1SsPx = anX+anOff1.x;
                int aYIm1SsPx = anY+anOff1.y;


                while (mod(aZ0,mNbByPix) != aPhase) aZ0++;
                int anOffset = Elise_div(aZ0,mNbByPix);

                // double aGlobCostGraphe = 0;
                double aGlobCostBasic  = 0;
                double aGlobCostCorrel = 0;

                for (int aZI=aZ0 ; aZI< aZ1 ; aZI+=mNbByPix)
                {
                        // double aZR = aZI * aStepPix;
                        double aCost = mAhDefCost;
                        if (OkIm0)
                        {
                            Pt2di aPIm1(aXIm1SsPx+anOffset,aYIm1SsPx);
                            if (anI1.IsOkErod(aPIm1.x,aPIm1.y))
                            {
                                if (DoGraphe)
                                {
                                     float aCostGr = 0;
                                     for (int aKS =0 ; aKS<aNbScale ; aKS++)
                                     {
                                         int aFlag0  = mDIF0[aKS][aPIm0.y][aPIm0.x];
                                         int aFlag1  = mDIF1[aKS][aPIm1.y][aPIm1.x];
                                         int aDFlag = aFlag0 ^ aFlag1;
                                         aCostGr +=  aDataPmsInit[aKS] * aCG->CostFlag(aDFlag);
                                     }
                                     aCost = aCostGr / aSomPmsInit;
                                     // aGlobCostGraphe = aCost;
                                     if (Verif) // Verification de cost Cennsus
                                     {

                                         Pt2dr aPRIm1 = Pt2dr(aPIm1) + Pt2dr(aPhase/double(mNbByPix),0);
                                         double aCostGrBas =CensusGraphePlein(aDataIm0,aPIm0,aDataIm1,aPRIm1.x,aPIm1.y,mCurSzVMax);
                                         // double aC3 = aCG->GainBasic(aBOI0.data(),aBOIC.data(),anOffset);
                                         // Peut pas forcer a 0, car interpol diff peut creer except une variation
                                         if ((ElAbs(aCostGrBas-aCost)>1e-4) )
                                         {
                                             std::cout << "Verfi Gr " << aCost << " " << aCostGrBas    << "==============================\n";
                                         }
                                     }
                                }
                                if (DoCensusBasic)
                                {
                                     U_INT2 * aFlag0 = aTabFlag0.Flag(aPIm0);
                                     U_INT2 * aFlag1 = aTabFlag1.Flag(aPIm1);
                                     aCost = aPondFlag->ValPonderDif(aFlag0,aFlag1);
                                     aGlobCostBasic = aCost;

                                     if (Verif)
                                     {
                                         Pt2dr aPRIm1 = Pt2dr(aPIm1) + Pt2dr(aPhase/double(mNbByPix),0);
                                         double aC2 = CensusBasic(aDataIm0,aPIm0,aDataIm1,aPRIm1.x,aPIm1.y,mCurSzVMax);
                                         if (ElAbs(aCost-aC2) > 1e-4)
                                         {
                                             std::cout << "Verfi Basic " << aCost <<  " "<< aC2 << "===================================\n";
                                         }

                                         double aC3 = CensusBasicCenter(aVBOI0[0]->data(),aVBOIC[0]->data(),anOffset,mCurSzVMax);
                                         if (ElAbs(aCost-aC3) > 1e-5)
                                         {
                                             std::cout << "Verfi Flag/Cen " << aCost <<  " "<< aC3 << "===================================\n";
                                             std::cout << aPIm0  << aTabFlag0.SzIm() << " " << aPIm1 << aTabFlag1.SzIm()<< "\n";
                                             getchar();
                                         }
                                     }
                                }
                                if (DoCensQuant)
                                {
                                    aCost =  Quick_MS_CensusQuant(aVBOI0,aVBOIC,anOffset,aVKImS,aVPds);
                                    aGlobCostCorrel = aCost; // ?? Pas sur utilite
  // return Quick_MS_CensusQuant(aVBOI1,aVBOI2,aPx2,aVV,aVPds);
                                }
                                if (DoCorrel)
                                {
/*
                                        aCost = MS_CorrelBasic_Center (aVBOI0,aVBOIC,anOffset,aVKImS,aVPds,mAhEpsilon,aModeMax);
*/
                                    aCost = Quick_MS_CorrelBasic_Center (aPIm0,aPIm1,aSom1,aSom11,aSomC,aSomCC,
                                                             aVBOI0,aVBOIC,anOffset,aVKImS,aVPds,mAhEpsilon,aModeMax
                                            );


                                    aGlobCostCorrel = aCost;

                                    if (Verif)
                                    {
                                        double aC3 = MS_CorrelBasic_Center (aVBOI0,aVBOIC,anOffset,aVKImS,aVPds,mAhEpsilon,aModeMax);

                                        if (ElAbs(aC3-aCost)> 1e-2)
                                        {
                                             std::cout << "??????Correl check census ?????? " << aCost << " " << aC3 << "\n";
                                             // ELISE_ASSERT(false,"Correl check census");
                                        }
                                    

                                         if (!CMS())
                                         {
                                             double aC2  = CorrelBasic_Center(aVBOI0[0]->data(),aVBOIC[0]->data(),anOffset,mCurSzVMax,mAhEpsilon);
                                             if (ElAbs(aC2-aCost) > 1e-5)
                                             {
                                                   std::cout << "COREELLL " << aCost << " " << aC2 << "\n";
                                                   ELISE_ASSERT(false,"Correl Check failed");
                                             }
                                         }
                                    }
                                }

                                if (DoMixte)
                                {
                                   if (aGlobCostCorrel>aSeuilHC)
                                   {
                                        aCost = aGlobCostCorrel;
                                   }
                                   else if (aGlobCostCorrel>aSeuilBC)
                                   {
                                        double aPCor = (aGlobCostCorrel-aSeuilBC) / (aSeuilHC-aSeuilBC);
                                        aCost =  aPCor * aGlobCostCorrel
                                                 + (1-aPCor) * aSeuilBC *  aGlobCostBasic;
                                   }
                                   else
                                   {
                                        aCost =  aSeuilBC *  aGlobCostBasic;
                                   }
                                }


// std::cout << " CCCcc " << aCost << "\n";



/*
                              aCost = CorrelBasic_ImInt(aDataIm0,aPIm0,aDataC,Pt2di(anX+anOff1.x+anOffset,anY+anOff1.y),mCurSzVMax,mAhEpsilon);
                              double aC2  = CorrelBasic_Center(aBOI0.data(),aBOIC.data(),anOffset,mCurSzVMax,mAhEpsilon);
                              double aC1 = CensusGraphe_ImInt(aDataIm0,aPIm0,aDataC,Pt2di(anX+anOff1.x+anOffset,anY+anOff1.y),mCurSzVMax);
                              double aC2 = CensusGraphe(aDataIm0,aPIm0,aDataIm1,anX+anOff1.x+aZR,anY+anOff1.y,mCurSzVMax,mAhEpsilon);
                              double aC3 = aCG->GainBasic(aBOI0.data(),aBOIC.data(),anOffset);

*/
                                aCost = mStatGlob->CorrelToCout(aCost);

                                aCost =  aDynCensusCost * aCost;
                            }
                        }
                        mSurfOpt->SetCout(Pt2di(anX,anY),&aZI,aCost);
                        anOffset++;
// std::cout << "ZZZZ " << aZI << " " << aCost << "\n";
                }
                for (int aK=0 ; aK<aNbBOI ; aK++)
                {
                    aVBOI0[aK]->AvanceY();
                    aVBOIC[aK]->AvanceY();
                }
                // aBOI0.AvanceY();
                // aBOIC.AvanceY();
             }
             for (int aK=0 ; aK<aNbBOI ; aK++)
             {
                 aVBOI0[aK]->AvanceX();
                 aVBOIC[aK]->AvanceX();
             }
             // aBOI0.AvanceX();
             // aBOIC.AvanceX();
        }
        DeleteAndClear(aVBOI0);
        DeleteAndClear(aVBOIC);
        delete aMomC;
    }


    DeleteAndClear(aVCG);
    delete aPondFlag;
    delete aMom1;

#ifdef CUDA_ENABLED
	GpGpuTools::Nvtx_RangePop();
#endif
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à  la mise en
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
associés au chargement,  à  l'utilisation,  à  la modification et/ou au
développement et à  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à  charger  et  tester  l'adéquation  du
logiciel à  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
