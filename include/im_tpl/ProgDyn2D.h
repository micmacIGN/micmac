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

#ifndef _ELISE_TplProgDyn2D
#define _ELISE_TplProgDyn2D

/*
Pt2di PBUG(547,832);
bool ToDebug(const Pt2di & aP)
{
    return
          (aP==PBUG)
       || (aP==PBUG+Pt2di(1,0))
       || (aP==PBUG+Pt2di(-1,0));
}
*/


typedef int tCostPrgD;
typedef enum
{
    eAvant = 0,
    eArriere = 1
} ePrgSens;


// template <class TargAux> class cTplCelOptProgDyn
// template <class TargAux> class cTplCelNapPrgDyn
// template <class TArg> class cProg2DOptimiser


//   une cTplCelOptProgDyn contient ce l'info necessaire pour faire les calcul
//  d'optimisation selon une direction
// 
//    une cTplCelNapPrgDyn contient juste l'info pour memoriser le cout agrege et
// le cout initial

// cProg2DOptimiser
//   Contient une tableau 3D de cTplCelNapPrgDyn
//   Contient une tableau 2D de cTplCelNapPrgDyn




template <class TargAux> class cTplCelOptProgDyn
{
     public :
        //cTplCelOptProgDyn() { }

        void InitCostSommet(int aCost)
        {
             mCostSommet = aCost;
        }

        // Initialisation des couts, fonction differente pour le premier sommet du
        // balayage et les autre
        void InitCumulPremierSommet(ePrgSens aSens)
        {
             mCostCum[aSens] = mCostSommet;
        }
        void InitCumulSommetStandard(ePrgSens aSens)
        {
             mCostCum[aSens] = int(1e9) ;
        }


        void UpdateCostOneArc
             (
                 const cTplCelOptProgDyn<TargAux> & aCel2,
                 ePrgSens aSens,
                 int aCostTrans
             )
        {
            ElSetMin
            (
                mCostCum[aSens],
                mCostSommet+aCostTrans+aCel2.mCostCum[aSens]
            );
        }

        tCostPrgD CostPassageForce() const
        {
            return mCostCum[eAvant] + mCostCum[eArriere] - mCostSommet;
        }

        TargAux & ArgAux() {return mAuxArg;}
        const TargAux & ArgAux() const {return mAuxArg;}

/*
        tCostPrgD GetCostSommet() const
        {
              return mCostSommet;
        }
*/
     private :
        //  cCelOptimProgDyn(const cCelOptimProgDyn &);

// 
        TargAux     mAuxArg;
        tCostPrgD   mCostCum[2];
        tCostPrgD   mCostSommet;

};


template <class TargAux> class cTplCelNapPrgDyn
{
      public :
            cTplCelNapPrgDyn() :
                mCostAgrege(0),
                mOwnCost(0)
            {
            }
            tCostPrgD OwnCost() const {return mOwnCost;}
            tCostPrgD CostAgrege() const {return mCostAgrege;}
            void SetOwnCost(double aVal)  {mOwnCost = (tCostPrgD)aVal;}
            void AddCostAgrege(double aVal)  {mCostAgrege += (tCostPrgD)aVal;}

            TargAux & ArgAux() {return mAuxArg;}
            const TargAux & ArgAux() const {return mAuxArg;}
       private :
            TargAux    mAuxArg;
            tCostPrgD  mCostAgrege;
            tCostPrgD  mOwnCost;
};

///  CONVENTIONS "Standard" sur les intervalles de nappes
///       aImZMin  <=    Z  < aImZMax

template <class TArg> class cProg2DOptimiser
{
     public :

             typedef cTplCelOptProgDyn<typename TArg::tArgCelTmp>  tCelOpt;
             typedef cTplCelNapPrgDyn<typename TArg::tArgNappe>    tCelNap;


             cProg2DOptimiser
             (
                   TArg  &         anArg,
                   Im2D_INT2       aZMin,
                   Im2D_INT2       aZMax,
                   int             aRab,
                   int             aMul
             ) :
                   mArg     (anArg),
                   mNappe   (aZMin,aZMax,aRab,aMul),
                   mRab     (aRab),
                   mMul     (aMul),
                   mTeta0   (0),
                   mSz      (aZMin.sz()),
                   mImZMin  (aZMin),
                   mTZMin   (mImZMin),
                   mImZMax  (aZMax),
                   mTZMax   (mImZMax),
                   mZMin    (aZMin.data()),
                   mZMax    (aZMax.data()),
                   mLMR     (mSz)
             {
             }


             cDynTplNappe3D<tCelNap> & Nappe() {return mNappe;}
             void DoOptim ( int aNbDir);
             double DMoyDir() const {return mDMoyDir;};
             void SetTeta0(double aTeta0){mTeta0=aTeta0;}
             // Pour recupere les donnees en sorties
             void TranfereSol(INT2**);
             const Pt2di  & Dir() const {return mDir;}
     private :

             void TransfertNapOpt(const std::vector<Pt2di> & aVP,bool VersOpt);
             void DoOneDirection(int aKDir);
             void BalayageOneLine(const std::vector<Pt2di> &);


              void BalayageOneSens
                   (
                        const std::vector<Pt2di> & aVP,
                        ePrgSens,
                        int aBegin,
                        int aEnd,
                        int anIncrem
                   );





            TArg &                                        mArg;
            cDynTplNappe2D<tCelOpt>                       mCels;
            cDynTplNappe3D<tCelNap>                       mNappe;
            int                                           mRab;
            int                                           mMul;
            double                                        mTeta0;
            Pt2di                                         mSz;
            Im2D_INT2                                     mImZMin;
            TIm2D<INT2,INT>                               mTZMin;
            Im2D_INT2                                     mImZMax;
            TIm2D<INT2,INT>                               mTZMax;
            signed short **                               mZMin;
            signed short **                               mZMax;
            cLineMapRect                                  mLMR;
            int                                           mNbDir;
            Pt2di                                         mDir;
            double                                        mDMoyDir;  // Dist moy entre deux point selon la direction
            int                                           mNbLine;
            std::vector<INT2>                             mLineZMin;
            std::vector<INT2>                             mLineZMax;
};

template <class TArg> void cProg2DOptimiser<TArg>::BalayageOneSens
                           (
                                 const std::vector<Pt2di> & aVP,
                                 ePrgSens aSens,
                                 int aBegin,
                                 int aEnd,
                                 int anIncrem
                           )
{
    {
       int aZMin = mLineZMin[aBegin]*mMul       ;
       int aZMax = mLineZMax[aBegin]*mMul + mRab;
       tCelOpt *aVCelOpt = mCels.Data()[aBegin];
       for (int aZ=aZMin ; aZ<aZMax ; aZ++)
       {
           aVCelOpt[aZ].InitCumulPremierSommet(aSens);
       }
    }

   for (int aNext = aBegin+anIncrem; aNext != aEnd; aNext += anIncrem)
   {
       {
          int aZMin = mLineZMin[aNext]*mMul       ;
          int aZMax = mLineZMax[aNext]*mMul + mRab;
          tCelOpt *aVCelOpt = mCels.Data()[aNext];
          for (int aZ=aZMin ; aZ<aZMax ; aZ++)
          {
              aVCelOpt[aZ].InitCumulSommetStandard(aSens);
          }
       }

        int aPrec = aNext-anIncrem;
        mArg.DoConnexion
        (
             aVP[aPrec],
             aVP[aNext],
             aSens,mRab,mMul,
             mCels.Data()[aPrec],mLineZMin[aPrec],mLineZMax[aPrec],
             mCels.Data()[aNext],mLineZMin[aNext],mLineZMax[aNext]
        );
   }
         
}

template <class TArg>  void cProg2DOptimiser<TArg>::TranfereSol(INT2** aSol)
{
    Pt2di aP;
    for (aP.x=0 ; aP.x<mSz.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y; aP.y++)
        {
             tCelNap * aVNap = mNappe.Data()[aP.y][aP.x];
             int aZ0 = mTZMin.get(aP)*mMul      ;
             int aZ1 = mTZMax.get(aP)*mMul+ mRab;

             tCostPrgD aBestCost = aVNap[aZ0].CostAgrege();
             int aBestZ = aZ0;
             for (int aZ=aZ0+1; aZ<aZ1; aZ++)
             {
                  tCostPrgD aCost = aVNap[aZ].CostAgrege();
                  if (aCost < aBestCost)
                  {
                       aBestCost = aCost;
                       aBestZ = aZ;
                  }
             }
             aSol[aP.y][aP.x] = aBestZ;
        }
    }
}


/*
    Transfere les donnees entre la nappe globale (3D) et la nappe temporaire (2D)
    Dans le sens 2D->3D, enleve le min
*/
template <class TArg>  void cProg2DOptimiser<TArg>::TransfertNapOpt(const std::vector<Pt2di> & aVP,bool VersOpt)
{
   for (int aKP=0; aKP<mNbLine ; aKP++)
   {
       
       Pt2di aP = aVP[aKP];
       tCelNap * aVNap = mNappe.Data()[aP.y][aP.x];
       tCelOpt *aVCelOpt = mCels.Data()[aKP];

       int aZMin = mLineZMin[aKP]*mMul       ;
       int aZMax = mLineZMax[aKP]*mMul + mRab;

       ELISE_ASSERT(aZMin<aZMax,"Empty Nap in cProg2DOptimiser<TArg>::TransfertNapOpt");
       
       tCostPrgD aMinCostF = (tCostPrgD)1e9;
       if (!VersOpt)
       {
          aMinCostF  = aVCelOpt[aZMin].CostPassageForce();
          for (int aZ= aZMin+1 ; aZ<aZMax ; aZ++)
          {
              ElSetMin(aMinCostF,aVCelOpt[aZ].CostPassageForce());
          }
       }
           

       for (int aZ=aZMin ; aZ<aZMax ; aZ++)
       {
            if (VersOpt)
            {
                aVCelOpt[aZ].InitCostSommet(aVNap[aZ].OwnCost());
                aVCelOpt[aZ].ArgAux().InitTmp(aVNap[aZ]);
            }
            else
            {
                 aVNap[aZ].AddCostAgrege(aVCelOpt[aZ].CostPassageForce()-aMinCostF);
            }
       }
   }
}


template <class TArg>  void cProg2DOptimiser<TArg>::BalayageOneLine(const std::vector<Pt2di> & aVP)
{
   mNbLine = (int)aVP.size();
   mLineZMin.clear();
   mLineZMax.clear();
   for (int aKP=0; aKP<mNbLine ; aKP++)
   {
         mLineZMin.push_back(mTZMin.get(aVP[aKP]));
         mLineZMax.push_back(mTZMax.get(aVP[aKP]));
   }
   mCels.Resize(VData(mLineZMin),VData(mLineZMax),mNbLine,mRab,mMul);

   TransfertNapOpt(aVP,true);
   BalayageOneSens(aVP,eAvant  ,        0,mNbLine, 1);
   BalayageOneSens(aVP,eArriere,mNbLine-1,     -1,-1);
   TransfertNapOpt(aVP,false);
}

template <class TArg> void cProg2DOptimiser<TArg>::DoOneDirection(int aKDir)
{
   double aTeta =  mTeta0 + (aKDir*PI)/mNbDir;
   mDir = Pt2di(Pt2dr::FromPolar(100.0,aTeta));
   mDMoyDir = average_euclid_line_seed(mDir);
   mArg.GlobInitDir(*this);


   mLMR.Init(mDir,Pt2di(0,0),mSz);

   const std::vector<Pt2di> * aVPt;
   while ((aVPt=mLMR.Next()))
          BalayageOneLine(*aVPt);

}

template <class TArg> void cProg2DOptimiser<TArg>::DoOptim(int aNbDir)
{
    // mArg.GlobalInitialisation(*this);
    mNbDir = aNbDir;

    for (int aKDir=0 ; aKDir<aNbDir ; aKDir++)
    {
         DoOneDirection(aKDir);
    }
}


// Pas ePrgDAgrProgressif
// PAs Px1MultRegul

class cOptimLabelBinaire
{
    public :

        // Les couts sont entre 0 et 1
        cOptimLabelBinaire(Pt2di aSz,double aDefCost,double aRegul);

        static cOptimLabelBinaire * CoxRoy(Pt2di aSz,double aDefCost,double aRegul);
        static cOptimLabelBinaire * ProgDyn(Pt2di aSz,double aDefCost,double aRegul); // 2 Do


        // 0.0 privilégie l'état 0 ; 1.0 privilégie l'état 1 ....
        void SetCost(Pt2di aP,double aCost);

        virtual Im2D_Bits<1> Sol() = 0;
        virtual ~cOptimLabelBinaire();

    protected :
        static U_INT1 ToCost(double aCost);

        Pt2di              mSz;
        Im2D_U_INT1        mCost;  // Memorise les couts entre 0 et 1
        TIm2D<U_INT1,INT>  mTCost;  // Memorise les couts entre 0 et 1
        double             mRegul;


};




#endif  //  _ELISE_TplProgDyn2D











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
