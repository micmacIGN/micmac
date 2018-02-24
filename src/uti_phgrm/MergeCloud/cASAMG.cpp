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


#include "MergeCloud.h"


//     void ComputeIncidGradProf();



cASAMG::cASAMG(cAppliMergeCloud * anAppli,cImaMM * anIma)  :
   mAppli     (anAppli),
   mPrm       (anAppli->Param()),
   mIma       (anIma),
   mNameIm    (anIma->mNameIm),
   // mStdN      (cElNuage3DMaille::FromFileIm(mAppli->NameFileInput(true,anIma,".xml"))),
   mStdN      (cElNuage3DMaille::FromFileIm(mAppli->MMIN()->NameFileXml(eTMIN_Depth,mNameIm))),
   mResol     (mStdN->ResolSolGlob()),
   mMasqN     (mStdN->ImDef()),
   mTMasqN    (mMasqN),
   mImProf    (mStdN->ImProf()),
   mImCptr    (1,1),
   mTCptr     (mImCptr),
   mSz        (mStdN->SzUnique()),
   mImIncid   (1,1),
   mTIncid    (mImIncid),
   mImQuality (mSz.x,mSz.y,eQC_Out),
   mTQual     (mImQuality),
   mImLabFin  (mSz.x,mSz.y,eLFNoAff),
   mTLabFin   (mImLabFin),
   mImEnvSup  (mSz.x,mSz.y),
   mTEnvSup   (mImEnvSup),
   mImEnvInf  (mSz.x,mSz.y),
   mTEnvInf   (mImEnvInf),
   mLutDecEnv (theNbValCompr),
   mDLDE      (mLutDecEnv.data()),
   mHisto     (mAppli->MaxValQualTheo()+1,0),
   mDH        (mHisto.data()),
   mMaxNivH   (-1),
   mSSIma     (mStdN->DynProfInPixel() *  mAppli->Param().ImageVariations().SeuilStrictVarIma()),
   mISOM      (StdGetISOM(anAppli->ICNM(),anIma->mNameIm,anAppli->Ori())),
   mNivSelected  (-1)
{
   mIsMAP = mAppli->IsInImageMAP(this);

   bool doComputeIncid = mAppli->Param().ComputeIncid().Val();
   if (doComputeIncid)
   {
       mImIncid.Resize(mSz);
       mTIncid = TIm2D<U_INT1,INT>(mImIncid);
       ComputeIncidGradProf();
    // ComputeIncidAngle3D();
   }
   double aPente = mAppli->Param().PenteRefutInitInPixel().Val();

   ELISE_COPY(select(mImQuality.all_pts(),mMasqN.in()),eQC_NonAff,mImQuality.out());
// InspectQual();
   ComputeIncidKLip(mMasqN.in_proj(),aPente   , eQC_GradFaibleC1);
   ComputeIncidKLip(mMasqN.in_proj(),aPente*2 , eQC_GradFort);
   ELISE_COPY
   (
       select
       (
            mMasqN.all_pts(),
            (mImQuality.in()==eQC_NonAff) && dilat_32(mMasqN.in_proj()==0,2*mPrm.DilateBord().Val())
       ),
       eQC_Bord,
       mImQuality.out()
   );
   
   

   ComputeSubset(mAppli->Param().NbPtsLowResume(),mLowRN);

   {
       Video_Win * aW = mAppli->Param().VisuGrad().Val() ? TheWinIm() : 0;
       if (aW)
       {
          aW->set_title(mIma->mNameIm.c_str());

          if (doComputeIncid)
          {
             ELISE_COPY
             (
                 mImIncid.all_pts(),
                 mImIncid.in(),
                 aW->ogray()
             );
             aW->clik_in();
          }
          else
          {
             InspectQual(true);
          }
       }
   }

   {
       Im2D_REAL4 aNoCompEnvSup(mSz.x,mSz.y);
       Im2D_REAL4 aNoCompEnvInf(mSz.x,mSz.y);
       if( (mAppli->Param().ModeMerge() == eQuickMac)
         ||  (mAppli->Param().ModeMerge() == eStatue)
         )
       {
           // std::string aNameEnvSup = mAppli->NameFileInput(true,mIma,"_Prof.tif","Max");
           // std::string aNameEnvInf = mAppli->NameFileInput(true,mIma,"_Prof.tif","Min");
           std::string aNameEnvSup = mAppli->MMIN()->NameFileProf(eTMIN_Max,mNameIm);
           std::string aNameEnvInf = mAppli->MMIN()->NameFileProf(eTMIN_Min,mNameIm);
           ELISE_COPY (aNoCompEnvSup.all_pts(),Tiff_Im(aNameEnvSup.c_str()).in(0),aNoCompEnvSup.out());
           ELISE_COPY (aNoCompEnvInf.all_pts(),Tiff_Im(aNameEnvInf.c_str()).in(0),aNoCompEnvInf.out());
             
       }
       else
       {
             ELISE_ASSERT(false,"Calcul envlop 2 do");
       }

       double aDiv = 2.0;
       double anExp = 4/5.0;
       Im1D_INT4 aLutCompr(256);
       ELISE_COPY(aLutCompr.all_pts(),Min(theNbValCompr-1,round_down(pow(FX/aDiv,anExp))),aLutCompr.out());
       ELISE_COPY(mLutDecEnv.all_pts(),round_up(aDiv*pow(FX+0.9999,1/anExp)),mLutDecEnv.out());

       // A conserver, permet d'inspecter compression/decompression
       if (0)
       {
           for (int aK=255 ; aK>=0 ; aK--)
           {
                 int * aLC =  aLutCompr.data();
                 std::cout << " K=" << aK << " Comp= " << aLC[aK] <<  " Dec=" << mDLDE[aLC[aK]] << "\n";
           }
       }

       Symb_FNum aIP (mStdN->ImProf()->in(0));
       Fonc_Num aCompEnvSup =  aLutCompr.in()[Max(0,Min(255,round_ni(aNoCompEnvSup.in()-aIP)))];
       Fonc_Num aCompEnvInf =  aLutCompr.in()[Max(0,Min(255,round_ni(aIP-aNoCompEnvInf.in())))];
       ELISE_COPY
       (
              aNoCompEnvSup.all_pts(),
              Virgule(aCompEnvInf,aCompEnvSup),
              Virgule(mImEnvInf.out(),mImEnvSup.out())
       );
      
       InspectEnv();
   }
}

double  cASAMG::InterioriteEnvlop(const Pt2di & aP,double aProfTest,double & aDeltaProf) const
{
   if ( (aP.x<0) || (aP.y<0) || (aP.x>=mSz.x) || (aP.y>=mSz.y)) return -1000;

   aDeltaProf = aProfTest - mImProf->GetR(aP);
   return (aDeltaProf>0) ?  (mTEnvSup.get(aP)-aDeltaProf) : (mTEnvInf.get(aP)+aDeltaProf);
}

const int cASAMG::theNbValCompr = 16;

void  cASAMG::InspectEnv()
{
    Video_Win * aW = mPrm.VisuEnv().Val() ? TheWinIm() : 0;
    if (! aW) return;
    ELISE_COPY
    (
        mImEnvSup.all_pts(),
        Min(255,mLutDecEnv.in()[mImEnvInf.in()]+mLutDecEnv.in()[mImEnvSup.in()])*mMasqN.in(),
        aW->ogray()
    );
    while (true)
    {
        Clik  aClk =  aW->clik_in();
        if (aClk._b!=1) return;

        Pt2di aP = round_ni(aClk._pt);

        std::cout << "INF " <<   mDLDE[mTEnvInf.get(aP)] << " SUP " << mDLDE[mTEnvSup.get(aP)] << "\n";
    }
}

void cASAMG::InspectQual(bool WithClik)
{
     Video_Win * aW = TheWinIm();
     if (! aW) return;

     aW->set_title(mIma->mNameIm.c_str());
     ELISE_COPY 
     (
          mImQuality.all_pts(),
          Min(255,mImQuality.in() * (255.0/mAppli->MaxValQualTheo())),
          aW->ogray()
     );
     ELISE_COPY 
     (
          select(mImQuality.all_pts(),mImQuality.in()==eQC_NonAff),
          P8COL::red,
          aW->odisc()
     );
     ELISE_COPY 
     (
          select(mImQuality.all_pts(),mImQuality.in()==eQC_Out),
          P8COL::yellow,
          aW->odisc()
     );

     bool Cont = WithClik;

     while (Cont)
     {
         Clik  aClk =  aW->clik_in();
         if (aClk._b==1)
         {
             std::cout << "Qual= " << eToString(eQualCloud(mTQual.get(round_ni(aClk._pt),0))) << aClk._pt << "\n";
         }
         else
         {
            Cont = false;
         }
     }
}


INT cASAMG::MaxNivH() const {return mMaxNivH;}
double cASAMG::QualOfNiv() const {return mQualOfNiv;}
int cASAMG::NbOfNiv() const {return mNbOfNiv;}
int cASAMG::NbTot() const {return mNbTot;}
bool  cASAMG::IsImageMAP() const {return mIsMAP;}
Pt2di cASAMG::Sz() const {return mSz;}
Video_Win *  cASAMG::TheWinIm() const {return mAppli->TheWinIm(this);}
double cASAMG::Resol() const {return mResol;}




cImaMM * cASAMG::IMM() {return  mIma;}

const cImSecOfMaster &  cASAMG::ISOM() const
{
   return mISOM;
}

void  cASAMG::AddCloseVois(cASAMG * anA)
{
   mCloseNeigh.push_back(anA);
}


bool  cASAMG::IsSelected() const {return mNivSelected >=0;}
int  cASAMG::NivSelected() const {return mNivSelected ;}


const cOneSolImageSec *  cASAMG::SolOfCostPerIm(double aCostPerIm)
{
   double aBestGain = -1e10;
   const cOneSolImageSec * aSol=0;
   for 
   (
        std::list<cOneSolImageSec>::const_iterator itS=mISOM.Sols().begin() ;
        itS !=mISOM.Sols().end() ;
        itS++
   )
   {
       double aGain = itS->Coverage() - aCostPerIm*itS ->Images().size();
       if (aGain > aBestGain)
       {
           aBestGain = aGain;
           aSol = & (*itS);
       }
   }
   return aSol;
}


 Im2D_Bits<1> MasqInterp(Pt2di aSzIn,Pt2di aSzOut,double aScale,Fonc_Num aFonc,double aSeuil)
{
    Im2D_U_INT1  aMasqBrd(aSzIn.x,aSzIn.y);
    TIm2D<U_INT1,INT>  aTMB(aMasqBrd);
    ELISE_COPY
    (
         aMasqBrd.all_pts(),
         aFonc,
         aMasqBrd.out()
    );

    
    Im2D_Bits<1> aMasqBrdFull(aSzOut.x,aSzOut.y);
    TIm2DBits<1> aTMBF(aMasqBrdFull);
    Pt2di aP;
    for (aP.x=0 ; aP.x<aSzOut.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzOut.y ; aP.y++)
        {
             Pt2dr aPR = Pt2dr(aP) / aScale;
             aTMBF.oset(aP,aTMB.getr(aPR,0)>=aSeuil);
        }
    }

    return aMasqBrdFull;
}


std::string  cASAMG::ExportMiseAuPoint()
{
    if (! IsSelected()) return"";

    // On genere les export a sous resol (surtout pour visualisation tempo)
    cXML_ParamNuage3DMaille aParam = mStdN->Params();

    Fonc_Num  aFOK = NFoncDilatCond
                     (
                         mImLabFin.in(eLFNoAff) == eLFMaster,
                         mImLabFin.in(eLFNoAff) != eLFNoAff,
                         true,
                         3
                      );
   aFOK = NFoncDilatCond
                     (
                         aFOK,
                         mImLabFin.in(eLFNoAff) != eLFNoAff,
                         false,
                         2
                      );


    if (0)
    {
       ELISE_COPY
       (
           select(mImLabFin.all_pts(),aFOK &&  mImLabFin.in()==eLFMasked),
           eLFBorder,
           mImLabFin.out()
       );
    }


    // std::string aNameMasq = mAppli->NameFileInput(true,mIma,"_Masq.tif","Test");
    // std::string aNameLab = mAppli->NameFileInput(true,mIma,"_Label.tif","Test");
    // std::string aNameXML = mAppli->NameFileInput(true,mIma,".xml","Test");


    std::string aNameMasq = mAppli->MMIN()->NameFileMasq(eTMIN_Merge,mNameIm);
    std::string aNameLab = mAppli->MMIN()->NameFileLabel(eTMIN_Merge,mNameIm);
    std::string aNameXML = mAppli->MMIN()->NameFileXml(eTMIN_Merge,mNameIm);

    // Tiff_Im::Create8BFromFonc(aNameMasq,mImLabFin.sz(),(mImLabFin.in()==eLFMaster)||(mImLabFin.in()==eLFBorder));
    Tiff_Im::Create8BFromFonc(aNameMasq,mImLabFin.sz(),mImLabFin.in()==eLFMaster);
    Tiff_Im::Create8BFromFonc(aNameLab,mImLabFin.sz(),mImLabFin.in());

    aParam.Image_Profondeur().Val().Masq() = NameWithoutDir(aNameMasq);
    MakeFileXML(aParam,aNameXML);

    std::string aComPly =  MM3dBinFile("Nuage2Ply") + "  " + aNameXML;
    if (mAppli->DoPlyCoul())
    {
       aComPly = aComPly 
                  + " "+QUOTE("Attr=" + mAppli->Dir()+mIma->mNameIm) 
                  + " RatioAttrCarte=" + ToString(mStdN->Params().SsResolRef().Val());
    }

    if (mAppli->HasOffsetPly())
    {
        aComPly  = aComPly + " Offs=" + ToString(mAppli->OffsetPly());
    }

    if (mAppli->Export64BPly())
    {
	aComPly = aComPly + " 64B=1";
    }
    else
    {
	aComPly = aComPly + " 64B=0";
    }

    if (mAppli->SzNormale() >0)
    {
         aComPly =    aComPly 
                    + " Normale=" + ToString(mAppli->SzNormale()) 
                    // + " NeighMask=" +  mAppli->NameFileInput(true,mIma->mNameIm,"_Masq.tif","Depth");
                    + " NeighMask=" +   mAppli->MMIN()->NameFileMasq(eTMIN_Depth,mNameIm);

    }
    /*
    else if (mAppli->NormaleByCenter())
    {
          aComPly = aComPly + " Center=true";
    }
    */
    // GIANG For Paul TS 23/11/2017 . export centre optique dans C3DC Nuage.
    else if (mAppli->NormaleByCenter())
    {
          aComPly = aComPly + " NormByC=2";
    }
    return aComPly;
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
