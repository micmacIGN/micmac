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

#ifndef _ELISE_NEW_ORI_H
#define _ELISE_NEW_ORI_H

#include "StdAfx.h"

class cNewO_OneIm;
class cNewO_CpleIm;
class cNewO_NameManager;
class cNewO_Appli;



template <const int TheNbPts,class Type>  class cFixedMergeTieP
{
     public :
       typedef cFixedMergeTieP<TheNbPts,Type> tMerge;
       typedef std::map<Type,tMerge *>     tMapMerge;

       cFixedMergeTieP() ;
       void FusionneInThis(cFixedMergeTieP<TheNbPts,Type> & anEl2,tMapMerge * Tabs);
       void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2);

        bool IsInit(int aK) const {return mTabIsInit[aK];}
        const Type & GetVal(int aK)    const {return mVals[aK];}
        bool IsOk() const {return mOk;}
        void SetNoOk() {mOk=false;}
        int  NbArc() const {return mNbArc;}
        void IncrArc() { mNbArc++;}
        int  NbSom() const ;
     private :
        void AddSom(const Type & aV,int aK);

        Type mVals[TheNbPts];
        bool  mTabIsInit[TheNbPts];
        bool  mOk;
        int   mNbArc;
};

template <const int TheNb,class Type> class cFixedMergeStruct
{
     public :
        typedef cFixedMergeTieP<TheNb,Type> tMerge;
        typedef std::map<Type,tMerge *>     tMapMerge;
        typedef typename tMapMerge::iterator         tItMM;

        void DoExport();
        const std::list<tMerge *> & ListMerged() const;


        void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2);
        cFixedMergeStruct();

        const Type & ValInf(int aK) const {return mEnvInf[aK];}
        const Type & ValSup(int aK) const {return mEnvSup[aK];}


     private :
        void AssertExported() const;
        void AssertUnExported() const;

        tMapMerge                           mTheMaps[TheNb];
        Type                                mEnvInf[TheNb];
        Type                                mEnvSup[TheNb];
        int                                 mNbSomOfIm[TheNb];
        std::vector<int>                    mStatArc;
        bool                                mExportDone;
        std::list<tMerge *>                 mLM;
};

typedef cFixedMergeStruct<2,Pt2dr> tMergeLPackH;
typedef cFixedMergeTieP<2,Pt2dr>   tMergeCplePt;
typedef std::list<tMergeCplePt *>  tLMCplP;
ElPackHomologue ToStdPack(const tMergeLPackH *,bool PondInvNorm,double PdsSingle=0.1);

ElPackHomologue PackReduit(const ElPackHomologue & aPack,int aNbInit,int aNbFin);


class cNewO_OneIm
{
    public :
            cNewO_OneIm
            (
                 cNewO_NameManager & aNM,
                 const std::string  & aName
            );

            CamStenope * CS();
            const std::string & Name() const;
            const cNewO_NameManager&  NM() const;
    private :
            cNewO_NameManager*  mNM;
            CamStenope *        mCS;
            std::string         mName;
};

class cNOCompPair
{
    public :
       cNOCompPair(const Pt2dr & aP1,const Pt2dr & aP2,const double & aPds);

       Pt2dr mP1;
       Pt2dr mP2;
       double mPds;
       Pt3dr  mQ1;
       Pt3dr  mQ2;
       Pt3dr  mQ2R;
       Pt3dr  mU1vQ2R;
};





class cNewO_CpleIm
{
    public :
          cNewO_CpleIm
          (
                cNewO_OneIm * aI1,
                cNewO_OneIm * aI2,
                tMergeLPackH *      aMergeTieP,
                ElRotation3D *      aTesSol,
                bool                Show
          );

          double ExactCost(const ElRotation3D & aRot,double aTetaMax) const;
    private :
          
       //======== Amniguity ====
            void CalcAmbig();
            void CalcSegAmbig();
            ElRotation3D  SolOfAmbiguity(double aTeta);

            Pt3dr CalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0);
            Pt3dr OneIterCalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0);
            Pt2dr ToW(const Pt2dr & aP) const;
            void ShowPack(const ElPackHomologue & aPack,int aCoul,double aRay);
            void ClikIn();


       //===================
          void  AddNewInit(const ElRotation3D & aR);
          double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2) const;


          double CostLinear(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const;
          double CostLinear(const ElRotation3D & aRot,const Pt3dr & aP1,const Pt3dr & aP2,double aTetaMax) const;

          void TestCostLinExact(const ElRotation3D & aRot);
          void AmelioreSolLinear(ElRotation3D  aRot,const std::string & aMes);
          ElRotation3D OneIterSolLinear(const ElRotation3D & aRot,std::vector<cNOCompPair> &,double & anErStd,double & aErMoy);


          double ExactCost
                 (Pt3dr & anI,const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const;



          cNewO_OneIm *     mI1;
          cNewO_OneIm *     mI2;
          tMergeLPackH *    mMergePH;
          ElRotation3D *    mTestC2toC1;
          ElPackHomologue   mPackPDist;
          ElPackHomologue   mPackPStd;
          Pt2dr             mPInfI1;
          Pt2dr             mPSupI1;
          ElPackHomologue   mPackStdRed;
          

     // Resolution lineraire
          int                      mNbCP;
          double                   mErStd;
          std::vector<cNOCompPair> mStCPairs;
          std::vector<cNOCompPair> mRedCPairs;
          L2SysSurResol            mSysLin5;
          L2SysSurResol            mSysLin2;
          L2SysSurResol            mSysLin3;
          bool                     mShow;
       

          ElRotation3D  mBestSol;
          double        mCostBestSol;
          bool          mBestSolIsInit;
          double        mBestErrStd;
          std::vector<double> mResidBest;
          std::vector<double> mCurResidu;

     // Ambiguite
          Pt3dr         mDirAmbig;
          ElSeg3D       mSegAmbig;
          Pt3dr         mIA;  // Intersetion
     // ===============================
          Video_Win *   mW;
          Pt2dr         mP0W;
          double        mScaleW;
};


class cNewO_NameManager
{
     public :
           cNewO_NameManager
           (
               const std::string  & aDir,
               const std::string  & anOri,
               const std::string  & PostTxt
           );
           CamStenope * CamOfName(const std::string & aName);
           ElPackHomologue PackOfName(const std::string & aN1,const std::string & aN2) const;
           const std::string & Dir() const;

           // 
           CamStenope * CamOriOfName(const std::string & aName,const std::string & anOri);

     private :
           cInterfChantierNameManipulateur * mICNM;
           std::string                       mDir;
           std::string                       mOriCal;
           std::string                       mPostHom;
};


template <const int TheNb> void NOMerge_AddPackHom
                           (
                                cFixedMergeStruct<TheNb,Pt2dr> & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera & aCam1,int aK1,
                                const ElCamera & aCam2,int aK2
                           );

template <const int TheNb> void NOMerge_AddAllCams
                           (
                                cFixedMergeStruct<TheNb,Pt2dr> & aMap,
                                std::vector<cNewO_OneIm *> aVI
                           );


class cCdtCombTiep
{
    public :
        typedef cFixedMergeTieP<2,Pt2dr> tMerge;
        cCdtCombTiep(tMerge * aM) ;
        Pt3dr NormQ1Q2();

        tMerge * mMerge;
        Pt2dr    mP1;
        double   mDMin;
        bool     mTaken;
        double   mPdsOccup;
        Pt3dr    mQ1;
        Pt3dr    mQ2;
        Pt3dr    mQ2Init;
};


class cNewO_CombineCple
{
    public :
         typedef cFixedMergeTieP<2,Pt2dr> tMerge;
         cNewO_CombineCple(const  cFixedMergeStruct<2,Pt2dr>  & aM,ElRotation3D * aTestSol);

    private :
          double CostOneArc(const Pt2di &);
          double CostOneBase(const Pt3dr & aBase);

          Pt2dr ToW(const Pt2dr &) const;
          void SetCurRot(const Pt3di & aP);
          void SetCurRot(const  ElMatrix<double> & aP);

          double K2Teta(int aK) const;
          int    PInt2Ind(const Pt3di  & aP) const;
          Pt3dr   PInt2Tetas(const Pt3di  & aP) const;

          double GetCost(const Pt3di  & aP) ;
          double  CalculCostCur();

          int               mCurStep;
          int               mNbStepTeta;
          ElMatrix<double>  mCurRot;
          Pt3di             mCurInd;
          Pt3dr             mCurTeta;
          Pt3dr             mCurBase;

          std::map<int,double>     mMapCost;
          std::vector<cCdtCombTiep> mVAllCdt;
          std::vector<cCdtCombTiep*> mVCdtSel;
          std::list<Pt2di>         mLArcs;

          Video_Win *                mW;
          double                     mScaleW;
          Pt2dr                      mP0W;
         
};

inline double AttenTetaMax(const double & aVal,const double & aVMax)
{
      if (aVMax<=0) return aVal;
      return  (aVal*aVMax) / (aVal + aVMax);
}





#endif // _ELISE_NEW_ORI_H

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
