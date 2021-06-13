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

   //============================= Partie traitement d'images =============================

// static int SeuilZC = 200;
// static int SzOpen32 = 9;

void ShowMaskAuto(TIm2D<INT2,INT> aTZ,TIm2DBits<1>  aTM,const std::string & aFile)
{
   Tiff_Im::Create8BFromFonc
   (
       aFile,
       aTZ.sz(),
       Virgule
       (
          mod(aTZ._the_im.in(),256),
          mod(aTZ._the_im.in(),256),
          108 +  aTM._the_im.in() * 48
       )
   );
}


// Plus ou moins completion K Liptsi inf

   //===============================================================================================
   //===============================================================================================
   //===============================================================================================
     
class cEmptyArgPrg2D
{
    public :


       double CostCroise(const cEmptyArgPrg2D &,const cEmptyArgPrg2D &  ) const {return 0;}
       void InitCpleRadiom(U_INT2,U_INT2) {}
       void InitVecMCP(const  std::vector<tMCPVal> &){}

       void InitTmp(const cTplCelNapPrgDyn<cEmptyArgPrg2D> &){}
};

class cTypeStdArgPrg2D
{
     public :

        typedef cEmptyArgPrg2D tArgCelTmp;  
        typedef cEmptyArgPrg2D tArgNappe;  

        typedef cEmptyArgPrg2D  tArgGlob;
};


//==================================================================
//==================================================================
//==================================================================

class cCpleValPrg2DCelNap
{
    public :
		
       void InitVecMCP(const  std::vector<tMCPVal> &){}
       void InitCpleRadiom(tCRVal aR1,tCRVal aR2)  
       {
            mR1 =aR1;
            mR2 =aR2;
       }

       tCRVal mR1;
       tCRVal mR2;

};

class cCpleValArgGlob;


class cCpleValArgGlob
{
     public :

          cCpleValArgGlob(const cCorrel_PonctuelleCroisee & aCPC) :
                mPdsCrois (aCPC.PdsCroisee())
          {
          }

          double mPdsCrois;
};

class cCpleValPrg2DTmp
{
    public :
       void InitTmp(const cTplCelNapPrgDyn<cCpleValPrg2DCelNap> & aCel)
       {
            mR1 = aCel.ArgAux().mR1;
            mR2 = aCel.ArgAux().mR2;
            mOK = (mR1!= ValUndefCple) && (mR2!= ValUndefCple);
       }

       cCpleValPrg2DTmp()  :
             mR1 (ValUndefCple),
             mR2 (ValUndefCple),
             mOK (false)
       {
       }
    
       tCRVal mR1;
       tCRVal mR2;
       bool   mOK;

       double N2() const {return ElSquare(double(mR1)) + ElSquare(double(mR2));}

       double CostCroise(const cCpleValPrg2DTmp &  aC2,const cCpleValArgGlob & anArg) const 
       {
           if ((!mOK) || (!aC2.mOK)) 
              return 1.0;
           double aRes = double(mR1) * aC2.mR2 - double(mR2) * aC2.mR1;

           aRes=  anArg.mPdsCrois * ElAbs(aRes / (N2() + aC2.N2()));
if (0) // (MPD_MM())
{
   std::cout << "CxRES = " << aRes  <<  " " <<  N2() + aC2.N2()  << "\n";
   getchar();
}
           return aRes;
       }
};


class cTypeClpeValArgPrg2D
{
     public :

        typedef cCpleValPrg2DTmp    tArgCelTmp;  
        typedef cCpleValPrg2DCelNap tArgNappe;  
        typedef cCpleValArgGlob     tArgGlob;
};

/*

       void InitTmp(const cTplCelNapPrgDyn<cEmptyArgPrg2D> &){}
*/
//==================================================================
//==================================================================
//==================================================================

template <class Type,const int NbV> class cTabValI1Prg2DCelNap
{
    public :

       void InitCpleRadiom(tCRVal aR1,tCRVal aR2)  { }
       // On laisse std::vector<Type>  pour que ca ne compile pas si !=  tMCPVal
       void InitVecMCP(const  std::vector<Type> & aV)
       {
            ELISE_ASSERT(NbV==aV.size(),"Incoh in cTabValI1Prg2DCelNap");
            memcpy(mVals,&(aV[0]),NbV*sizeof(Type));
       }

	   // visual does not allow arrays of size 0
       Type mVals[NbV==0?1:NbV];

};


class cTabValArgGlob
{
     public :

          cTabValArgGlob(const cMultiCorrelPonctuel & aMCP, const int & aValOut,double aDefRes) :
                mPdsCrois ((aMCP.PdsCorrelCroise() ) / TheDynMCP),
                mDefRes   (aMCP.PdsCorrelCroise() * aDefRes),
                mValOut   (aValOut)
          {
          }

          double mPdsCrois;
          double mDefRes;
          int    mValOut;
};

template <class Type,const int NbV> class cTabValI1Prg2DTmp
{
    public :
       void InitTmp(const cTplCelNapPrgDyn< cTabValI1Prg2DCelNap<Type,NbV> >  & aCel)
       {
            memcpy(mVals,aCel.ArgAux().mVals,NbV);
       }

       cTabValI1Prg2DTmp()  
       {
       }
	   
	   // visual does not allow arrays of size 0
       Type mVals[NbV==0?1:NbV];
    

       double CostCroise(const cTabValI1Prg2DTmp<Type,NbV> &  aC2,const cTabValArgGlob & anArg) const 
       {
           int  aRes = 0;
           int  aNbNN = 0;
           for (int aK=0 ; aK< NbV ; aK++)
           {
               Type aV1 = mVals[aK];
               Type aV2 = aC2.mVals[aK];

// std::cout << "V1V2 " << int(aV1) << " " << int(aV2) << "\n";
               if ((aV1!=anArg.mValOut) && (aV2!=anArg.mValOut)) 
               {
                  aRes += ElAbs(aV1-aV2) * MCPMulCorel;
                  aNbNN++;
               }
           }

// std::cout << "RRRR = " << aRes << "\n";

           if (aNbNN != 0)
           {
// std::cout << "RESS-NN " << (aRes * anArg.mPdsPonct) / aNbNN  << " " << anArg.mDefRes <<"\n";
              return (aRes * anArg.mPdsCrois) / aNbNN;
           }

// std::cout << "RESNUl " << anArg.mDefRes << "\n";

           return anArg.mDefRes;
       }
};


template <class Type,const int NbV> class cTypeTabValArgPgr2D    

{
     public :

        typedef cTabValI1Prg2DTmp<Type,NbV>               tArgCelTmp;  
        typedef cTabValI1Prg2DCelNap<Type,NbV>            tArgNappe;  
        typedef cTabValArgGlob                            tArgGlob;
};

//==================================================================
//==================================================================
//==================================================================



              //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


template <class TypeArg> class cMMNewPrg2D : public cSurfaceOptimiseur
{
     public :
      // Pre-requis (template instanciation)  pour  cProg2DOptimiser
        typedef typename TypeArg::tArgCelTmp tArgCelTmp;  
        typedef typename TypeArg::tArgNappe  tArgNappe;  
        typedef typename TypeArg::tArgGlob   tArgGlob;  

        // Pas pre-requis mais aide
        typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
        typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;

        void DoConnexion
             (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Ouput,int aOutZMin,int aOutZMax
             );

        void GlobInitDir(cProg2DOptimiser<cMMNewPrg2D> &);
        //  void GlobalInitialisation(cProg2DOptimiser<cMMNewPrg2D> &);

      // Pre-requis (interface virtuelle) pour cSurfaceOptimiseur
        void Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel);
        void Local_SolveOpt(Im2D_U_INT1);

        void Local_SetCpleRadiom(Pt2di aPTer,int * aPX,tCRVal aR1,tCRVal aR2);
        void Local_VecMCP(Pt2di aPTer,int * aPX,const  std::vector<tMCPVal> &);

     //====================  End Pre Requis =================



        cMMNewPrg2D
        (
                const tArgGlob &        anArgGlob,
                cAppliMICMAC &    anAppli,
                const cModulationProgDyn & aPrgD,
                const cEtapeProgDyn &      anEPG,
                cLoadTer&               aLT,
                const cEquiv1D &        anEqX,
                const cEquiv1D &        anEqY,
                int                     aMul,
                bool                    EtiqImage,
                const cEtiqBestImage  * anEBI
        );
		~cMMNewPrg2D();


      private :


        tArgGlob                mArgGlob;
        const cAppliMICMAC &    mAppli;
        cModulationProgDyn      mMod;
        cEtapeProgDyn           mEPG;
        const cEtapeMecComp &   mEtape;
        int                     mNumNap;  // Pour l'instant seult zero, mais a voir ...
        const cOneNappePx &     mONP;


        bool mHasMask;
        cArgMaskAuto *          mArgMaskAuto;
        bool              mEtiqImage;
        int               mMulZ;
        int               mCostChImage;
        cProg2DOptimiser<cMMNewPrg2D>   *mPrg2D;
        TIm2DBits<1>                    *mTMaskTer;
        cDynTplNappe3D<tCelNap>		    *mNap;
        tCelNap ***                     mCelsNap;
        
        int mMaxJumpGlob;
        int mMaxJumpDir;
        double mMaxPente;

   //  Couts 
        std::vector<int>  mCostJump;
        int               mCostDefMasked;
        int               mCostTransMaskNoMask;
        int               mCostOut;


        
        
};

       // TIm2DBits<1>    aTMaskTer(mLTCur->ImMasqTer());


template <class Type> cMMNewPrg2D<Type>::cMMNewPrg2D
(
    const tArgGlob &        anArgGlob,
    cAppliMICMAC &    anAppli,
    const cModulationProgDyn & aModPrg,
    const cEtapeProgDyn &      anEPG,
    cLoadTer&               aLT,
    const cEquiv1D &        anEqX,
    const cEquiv1D &        anEqY,
    int                     aMul,
    bool                    EtiqImage,
    const cEtiqBestImage  * anEBI
) :
   cSurfaceOptimiseur(anAppli,aLT,1e4,anEqX,anEqY,EtiqImage,true),
   mArgGlob (anArgGlob),
   mAppli   (anAppli),
   mMod     (aModPrg),
   mEPG     (anEPG),
   mEtape   (*(mAppli.CurEtape())),
   mNumNap  (mEtape.NumSeuleNapEp()),
   mONP     (mLTCur->KthNap(mNumNap)),
   mHasMask (mMod.ArgMaskAuto().IsInit()),
   mArgMaskAuto (mMod.ArgMaskAuto().PtrVal()),
   mEtiqImage    (EtiqImage),
   mMulZ         (aMul),
   mCostChImage  ( anEBI ? CostR2I(anEBI->CostChangeEtiq()) :0),
   mPrg2D				 (NULL),
   mTMaskTer			 (NULL),
   mNap	                 (NULL),
   mCostDefMasked        (0),
   mCostTransMaskNoMask  (0),
   mCostOut              (CostR2I(50.0))
{
	// NO_WARN
	mPrg2D		= new cProg2DOptimiser<cMMNewPrg2D>( *this, mONP.mImPxMin, mONP.mImPxMax, (mHasMask ? 1 : 0), mMulZ );
	mTMaskTer	= new TIm2DBits<1>( mLTCur->ImMasqTer() );
	mNap		= &( mPrg2D->Nappe() );
	mCelsNap	= mNap->Data();

    mPrg2D->SetTeta0(mEPG.Teta0().Val());
    mMaxJumpGlob  = MaxJump (mNap->IZMin(),mNap->IZMax());
    mMaxPente     = mMod.Px1PenteMax().Val()/ mEtape.KPx(mNumNap).ComputedPas();

    for (int aKJ=0 ; aKJ<=mMaxJumpGlob+1 ; aKJ++)
    {
        mCostJump.push_back(CostR2I(CostTransEnt(aKJ,0)));
        // std::cout << "COST " << aKJ << " " << mCostJump[aKJ] << "\n";
    }
}

template <class Type> cMMNewPrg2D<Type>::~cMMNewPrg2D(){
	if ( mPrg2D!=NULL ) delete mPrg2D;
	if ( mTMaskTer!=NULL ) delete mTMaskTer;
}

template <class Type> void cMMNewPrg2D<Type>::DoConnexion
     (
          const Pt2di & aPIn, const Pt2di & aPOut,
          ePrgSens aSens,int aRab,int aMul,
          tCelOpt*Input,int aInZMin,int aInZMax,
          tCelOpt*Ouput,int aOutZMin,int aOutZMax
     )
{
   bool okPIn  = ( mTMaskTer->get(aPIn)!=0 );
   bool okPOut = ( mTMaskTer->get(aPOut)!=0 );

   bool  okInOut = (okPIn && okPOut);


   if (okInOut || (!mHasMask))
   {
       for (int aZOut = aOutZMin; aZOut<aOutZMax ; aZOut++)
       {
           int aDZMin,aDZMax;
           if (mHasMask)
           {
              BasicComputeIntervaleDelta
              (
                 aDZMin,aDZMax,
                 aZOut,mMaxJumpDir,
                 aInZMin,aInZMax
              );
           }
           else
           {
              ComputeIntervaleDelta
              (
                 aDZMin,aDZMax,
                 aZOut,mMaxJumpDir,
                 aOutZMin,aOutZMax,
                 aInZMin,aInZMax
              );
           }

           if (mEtiqImage)
           {
               // Transition de changement d'etiquette , a Z constant
               if ((aDZMin<=0) && (aDZMax>=0))
               {
                    int aZchEt = aZOut;

                    int aZOut0 = aZOut * mMulZ;
                    int aZOut1 = aZOut0 + mMulZ;
                    int aZIn0 = aZchEt * mMulZ;
                    int aZIn1 = aZIn0 + mMulZ;

                    for (int aZEout = aZOut0 ; aZEout < aZOut1 ; aZEout++)
                    {
                        for (int aZEin = aZIn0 ; aZEin < aZIn1 ; aZEin++)
                        {
                             
                             Ouput[aZEout].UpdateCostOneArc(Input[aZEin],aSens,mCostChImage);
                        }
                    }
               }
 
               // Transition a etiquette constante, Z variable
               for (int aDZ = aDZMin; aDZ<=aDZMax ; aDZ++)
               {
                   int aZOut0 = aZOut * mMulZ;
                   int aZIn0  = (aZOut+ aDZ) * mMulZ;
                   int aCost = mCostJump[ElAbs(aDZ)];
                   for (int anEt = 0 ; anEt < mMulZ ; anEt++)
                       Ouput[aZOut0+anEt].UpdateCostOneArc(Input[aZIn0+anEt],aSens,aCost);
               }

           }
           else
           {
              tCelOpt & aCelOut = Ouput[aZOut];
 
              for (int aDZ = aDZMin; aDZ<=aDZMax ; aDZ++)
              {
                  ELISE_ASSERT((aZOut>=aOutZMin)&&(aZOut<aOutZMax),"Incoh Out in cMMNewPrg2D::DoConnexion");
                  ELISE_ASSERT(((aZOut+aDZ)>=aInZMin)&&((aZOut+aDZ)<aInZMax),"Incoh In in cMMNewPrg2D::DoConnexion");

                  int aCost = mCostJump[ElAbs(aDZ)];

                  tCelOpt & aCelIn = Input[aZOut+aDZ];

                  if (okInOut)
                  {

if (0) // (MPD_MM())
{
    double  aCR=  aCelIn.ArgAux().CostCroise(aCelOut.ArgAux(),mArgGlob);
    int aCI=  CostR2I(aCR);
    std::cout << "CCciii " <<  aCR << " " << aCI << "\n";
}


                      aCost += CostR2I(aCelIn.ArgAux().CostCroise(aCelOut.ArgAux(),mArgGlob));
                  }

                  aCelOut.UpdateCostOneArc(aCelIn,aSens,aCost);
              }
           }
       }
    }

   if (mHasMask)
   {
       int aMulOutZMin = aOutZMin* mMulZ;
       int aMulOutZMax = aOutZMax* mMulZ;
       int aMulInZMin = aInZMin* mMulZ;
       int aMulInZMax = aInZMax* mMulZ;

       if (okPOut)
       {
          for (int aZOut = aMulOutZMin; aZOut<aMulOutZMax ; aZOut++)
          {
              Ouput[aZOut].UpdateCostOneArc(Input[aMulInZMax],aSens,mCostTransMaskNoMask);
          }
       }
       Ouput[aMulOutZMax].UpdateCostOneArc(Input[aMulInZMax],aSens,0);

       if (okPIn)
       {
          for (int aZIn = aMulInZMin; aZIn<aMulInZMax ; aZIn++)
          {
              Ouput[aMulOutZMax].UpdateCostOneArc(Input[aZIn],aSens,mCostTransMaskNoMask);
          }
       }
   }
 
}

template <class Type>  void cMMNewPrg2D<Type>::GlobInitDir(cProg2DOptimiser<cMMNewPrg2D> & aPrg)
{
     int aJump    = round_ni(mMaxPente*aPrg.DMoyDir());
     mMaxJumpDir  = ElMax(1,aJump);
}

template <class Type> void cMMNewPrg2D<Type>::Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel)
{
   int aZ = aPX[0];

   if (mEtiqImage)
   {
       aZ =  aZ * mMulZ + aLabel;
   }

   mCelsNap[aPTer.y][aPTer.x][aZ].SetOwnCost(CostR2I(aCost));
}



template <class Type> void cMMNewPrg2D<Type>::Local_SetCpleRadiom(Pt2di aPTer,int * aPX,tCRVal aR1,tCRVal aR2)
{
   mCelsNap[aPTer.y][aPTer.x][aPX[0]].ArgAux().InitCpleRadiom(aR1,aR2);
}
template <class Type> void cMMNewPrg2D<Type>::Local_VecMCP(Pt2di aPTer,int * aPX,const  std::vector<tMCPVal> & aVec)
{
   mCelsNap[aPTer.y][aPTer.x][aPX[0]].ArgAux().InitVecMCP(aVec);
}



void CombleTrouPrgDyn
     (
         const cModulationProgDyn & aPrgD,
         Im2D_Bits<1>  aMaskCalc,
         Im2D_Bits<1>  aMaskTer,
         Im2D_INT2     aImZ
     )
{
    const cArgMaskAuto *  aArgMaskAuto = aPrgD.ArgMaskAuto().PtrVal();
    if (! aArgMaskAuto) return;
    int aNbOpen = aArgMaskAuto->SzOpen32().Val();
    if (aNbOpen!=0)
    {
       ELISE_COPY
       (
            aMaskCalc.all_pts(),
            open_32(aMaskCalc.in(1),aNbOpen),
            aMaskCalc.out()
       );
    }

    TIm2DBits<1>    aTMaskCalc(aMaskCalc);
       // filtrage des composantes connexes
    FiltrageCardCC(true,aTMaskCalc,1,0,aArgMaskAuto->SeuilZC().Val());
    FiltrageCardCC(true,aTMaskCalc,0,1,aArgMaskAuto->SeuilZC().Val());

    // TIm2D<INT2,INT>   aTImRes(mImRes[mNumNap]);
    TIm2DBits<1>    aTMaskTer(aMaskTer);

       // aTMask    --> masque de non correlation
       // aTMaskTer --> masque terrain
       //ComplKLipsParLBas(aTMaskTer.Im(),aTMask.Im(),mImRes[mNumNap],1.0);
    ComplKLipsParLBas(aTMaskCalc.Im(),aTMaskTer.Im(),aImZ,1.0);
/*
*/
}


template <class Type>  void cMMNewPrg2D<Type>::Local_SolveOpt(Im2D_U_INT1 aImCor)
{
    // double AmplifKL = 100.0;
    if (mHasMask)
    {
        const cArgMaskAuto & anAMA  = mMod.ArgMaskAuto().Val();
        // AmplifKL = anAMA.AmplKLPostTr().Val();
        mCostDefMasked = CostR2I(mAppli.CurCorrelToCout(anAMA.ValDefCorrel()));

// std::cout << "COST DEF MASKE " << mAppli.CurCorrelToCout(anAMA.ValDefCorrel()) << " " << mCostDefMasked << "\n";
        mCostTransMaskNoMask = CostR2I(anAMA.CostTrans());

// std::cout << "mCostTransMaskNoMask " << mCostTransMaskNoMask << "\n";

        Im2D_INT2  aIZMax = mNap->IZMax();
        Im2D_INT2  aIZMin = mNap->IZMin();
        Pt2di aSz = aIZMax.sz();
        INT2 ** aDZmax = aIZMax.data();
        INT2 ** aDZmin = aIZMin.data();
        TIm2DBits<1> aTMask(mLTCur->ImMasqTer());


        // Initialisation des cout d'etat de non correl
        // les point en dehors du masque sont + ou - forces a etre en non correl (via mCostOut)
        Pt2di aP;
        for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
        {
            for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
            {
                tCelNap * aVCel = mCelsNap[aP.y][aP.x];
                int aZMax = aDZmax[aP.y][aP.x] * mMulZ;
                // Si on est en dehors du masque , on force a etre en etat
                // de non correl, en donnant une valeur 0 a cet etat et tres grande
                // aux autres
                if (!aTMask.get(aP))
                {
                   int aZMin = aDZmin[aP.y][aP.x] * mMulZ;
                   for (int aZ=aZMin ; aZ<aZMax ; aZ++)
                   {
                       aVCel[aZ].SetOwnCost(mCostOut);
                   }
                   aVCel[aZMax].SetOwnCost(0);
                }
                else
                {
                   aVCel[aZMax].SetOwnCost(mCostDefMasked);
                }
            }
        }
    }

   mPrg2D->DoOptim(mEPG.NbDir().Val());
   // mDataImRes[0][y][x]
   mPrg2D->TranfereSol(mDataImRes[mNumNap]);

   Im2D_INT2  aIZMax = mNap->IZMax();
   Pt2di aSz = aIZMax.sz();

   Im2D_U_INT1 aImEtiq(aSz.x,aSz.y);

   if (mEtiqImage || mDoFileCorrel)
   {
       Pt2di aP;
       INT2 **         aRes = mDataImRes[mNumNap];
       for (aP.x=0; aP.x<aSz.x ; aP.x++)
       {
           for (aP.y =0 ; aP.y < aSz.y ; aP.y++)
           {
                int aVRes = aRes[aP.y][aP.x];
                
                if (mEtiqImage)
                {
                    aRes[aP.y][aP.x] = Elise_div(aVRes,mMulZ);
                    aImEtiq.SetI(aP,mAppli.NumImAct2NumImAbs(mod(aVRes,mMulZ)+1));
                }
                if (mDoFileCorrel)
                {
                   int aCostI = mCelsNap[aP.y][aP.x][aVRes].OwnCost();
                   int aCorI = CostI2CorExport(aCostI);
                   if (! mLTCur->IsInMasq(aP)) 
                      aCorI = 0;
                   aImCor.SetI(aP,aCorI);
                }
           }
      }
                 // int aVRes = aRes[aP.y][aP.x];
// mMulZ
   }


// std::cout << "AAAAAAAAAAAAAAAAAaa " << AmplifKL << "\n"; getchar();

   //  Filtrage du masque
   if (mHasMask)
   {
       mMaskCalcDone = true;
       mMaskCalc = Im2D_Bits<1>(aSz.x,aSz.y);

       TIm2D<INT2,INT> aTZMax(aIZMax);
       TIm2DBits<1>    aTMask(mMaskCalc);
       INT2 **         aRes = mDataImRes[mNumNap];
       Pt2di aP;

       // Im2D<INT2,INT>  aZEnvSup(aSz.x,aSz.y,0);
       // aZEnvSup.dup(mImRes[mNumNap]);
       // TIm2D<INT2,INT> aTZES(aZEnvSup);
       
       for (aP.x=0; aP.x<aSz.x ; aP.x++)
       {
           for (aP.y =0 ; aP.y < aSz.y ; aP.y++)
           {
                 int aVRes = aRes[aP.y][aP.x];
                 bool NoVal = (aTZMax.get(aP) == aVRes);
                 aTMask.oset
                 (
                       aP,
                       (!NoVal)  && ( mLTCur->IsInMasq(aP))
                 );
                 // aTZES.oset(aP,NoVal?SHRT_MIN:aVRes);
                 if (NoVal) 
                 {
                     aImEtiq.SetI(aP,0);
                     if (mDoFileCorrel)
                     {
                        aImCor.SetI(aP,1);
                     }
                 }
           }
       }
       CombleTrouPrgDyn(mMod,mMaskCalc,mLTCur->ImMasqTer(),mImRes[mNumNap]);



       if (0)
       {
          Video_Win aW = Video_Win::WStd(aSz,1);
          // ELISE_COPY(aW.all_pts(),mMaskCalc.in() + 2*mLTCur->ImMasqTer().in() ,aW.odisc());
          ELISE_COPY(aW.all_pts(),P8COL::yellow ,aW.odisc());
          ELISE_COPY(select(aW.all_pts(),mLTCur->ImMasqTer().in()),P8COL::red ,aW.odisc());
          getchar();
          ELISE_COPY(select(aW.all_pts(),mMaskCalc.in()),P8COL::blue ,aW.odisc());
          getchar();
       }

   }

   if (mEtiqImage)
   {
        // Video_Win aW = Video_Win::WStd(aSz,1);
        // ELISE_COPY(aW.all_pts(),aImEtiq.in(),aW.odisc());
        // getchar();
        Tiff_Im aTifEt = mAppli.CurEtape()->FileRes(GenIm::u_int1,"Etiq");

        mAppli.SauvFileChantier(aImEtiq.in(),aTifEt);
   }
}


cSurfaceOptimiseur * cSurfaceOptimiseur::AllocNewPrgDyn
                     (
                                     cAppliMICMAC &    aAppli,
                                     cLoadTer&         aLT,
                                     const cModulationProgDyn & aPrgD,
                                     const cEtapeProgDyn &      anEPG,     
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                     )
{
   const cCorrelAdHoc * aCAH = aAppli.CAH();
   int 	aMulImage = 1;
   bool EtiqImage = false;
   const cEtiqBestImage * anEBI = aAppli.EBI();

   if (anEBI)
   {
      EtiqImage = true;
      aMulImage = ElMax(1,aAppli.NbVueAct()-1);
   }

   if (aCAH)
   {
// if (MPD_MM()) { std::cout << "AAAAAAAAAAAAAa\n"; getchar(); }
        if (aCAH->Correl_PonctuelleCroisee().IsInit())
        {
// if (MPD_MM()) { std::cout << "BBBBBBB\n"; getchar(); }
            ELISE_ASSERT(! EtiqImage,"Incompatibilite  : PonctuelleCroisee / EtiqImage");
            cCpleValArgGlob anArg(aCAH->Correl_PonctuelleCroisee().Val());
            return new cMMNewPrg2D<cTypeClpeValArgPrg2D>(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
        }
        else if (aCAH->MultiCorrelPonctuel().IsInit())
        {
             const cMultiCorrelPonctuel & aMCP = aCAH->MultiCorrelPonctuel().Val();
             ELISE_ASSERT(! EtiqImage,"Incompatibilite  : MultiCorrelPonctuel / EtiqImage");
             cTabValArgGlob anArg(aMCP,ValUndefCPONT,aMCP.DefCost().Val());


             switch (aAppli.NbVueAct()-1)
             {
                  case 0 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,0> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 1 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,1> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 2 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,2> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 3 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,3> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 4 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,4> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 5 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,5> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 6 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,6> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 7 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,7> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 8 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,8> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 9 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,9> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
                  case 10 :
                     return new  cMMNewPrg2D<cTypeTabValArgPgr2D<tMCPVal,10> >(anArg,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);
                  break;
             }
             std::cout << "Nb Image " << aAppli.NbVueAct()  << "\n";
             ELISE_ASSERT(false,"Too much image in MultiCorrelPonctuel");
        }
   }


   cEmptyArgPrg2D anEA;
   return new cMMNewPrg2D<cTypeStdArgPrg2D>(anEA,aAppli,aPrgD,anEPG,aLT,anEqX,anEqY,aMulImage,EtiqImage,anEBI);

   // return new cMMNewPrg2D<cTypeClpeValArgPrg2D>(aAppli,aPrgD,anEPG,aLT,anEqX,anEqY);
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
