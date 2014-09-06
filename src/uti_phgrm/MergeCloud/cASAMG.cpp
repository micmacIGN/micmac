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


class cCalcPlan : public cCC_NoActionOnNewPt
{
    public  :
       cCalcPlan(int aNbMaxStep,L2SysSurResol & aSys,const cRawNuage & aRN);

       void OnNewStep() { mNbStep++;}
       void  OnNewPt(const Pt2di & aP) 
       {
             mVPts.push_back(aP);
             Pt3dr aP3 = mRN->GetPt(aP);
             mVP3.push_back(aP3);
             
       }
       bool  StopCondStep() {return mNbStep>mNbMaxStep;}
       bool ValidePt(const Pt2di &){return true;}

    // public :
       L2SysSurResol * mSys;
       const cRawNuage * mRN;
       int mNbStep;
       int mNbMaxStep;
       std::vector<Pt2di> mVPts;
       std::vector<Pt3dr> mVP3;
};

cCalcPlan::cCalcPlan(int aNbMaxStep,L2SysSurResol & aSys,const cRawNuage & aRN) :
   mSys       (&aSys),
   mRN        (&aRN),
   mNbStep    (0),
   mNbMaxStep (aNbMaxStep)
{
    mSys->Reset();
}
 



cASAMG::cASAMG(cAppliMergeCloud * anAppli,cImaMM * anIma)  :
   mAppli     (anAppli),
   mIma       (anIma),
   mStdN      (cElNuage3DMaille::FromFileIm(mAppli->NameFileInput(anIma,".xml"))),
   mImCptr    (Im2D_U_INT1::FromFileStd(mAppli->NameFileInput(anIma,"CptRed.tif"))),
   mTCptr     (mImCptr),
   mSz        (mImCptr.sz()),
   mImIncid   (mSz.x,mSz.y),
   mTIncid    (mImIncid)
{
   
bool pAramV4 = false;
int  pAramDistCC = 3;
double pAramDynAngul = 100.0;
   
   double aSeuilNbPts = 2 * (1+2*pAramDistCC);
   cRawNuage   aRN = mStdN->GetRaw();
   Pt2di aP0;
   TIm2DBits<1> aTDef(mStdN->ImDef());

   Im2D_Bits<1> aMasqTmp = ImMarqueurCC(mSz);
   TIm2DBits<1> aTMasqTmp(aMasqTmp);

   L2SysSurResol aSys(3);

   // CamStenope * aCam = mStdN->Cam();
   // Pt3dr aC0 = aCam->PseudoOpticalCenter();


   for (aP0.x=0 ; aP0.x<mSz.x ; aP0.x++)
   {
       for (aP0.y=0 ; aP0.y<mSz.y ; aP0.y++)
       {
            double Angle = 1.5;
            if (aTDef.get(aP0))
            {
                cCalcPlan  aCalcPl(pAramDistCC,aSys,aRN);
                int aNb = OneZC(aP0,pAramV4,aTMasqTmp,1,0,aTDef,1,aCalcPl);
                ResetMarqueur(aTMasqTmp,aCalcPl.mVPts);
                if (aNb >= aSeuilNbPts)
                {
                    cElPlan3D aPlan(aCalcPl.mVP3,0);
                    // ElSeg3D aSeg(aC0,mStdN->GetPt(aP0));
                    ElSeg3D aSeg =  mStdN->FaisceauFromIndex(Pt2dr(aP0));
                    double aScal = scal(aPlan.Norm(),aSeg.TgNormee());
                    if (aScal<0) aScal = - aScal;
                    Angle = acos(ElMin(1.0,ElMax(-1.0,aScal)));
                }
            }
            mTIncid.oset(aP0,ElMin(255,ElMax(0,round_ni(Angle*pAramDynAngul))));
       }
   }


   if (1)
   {
      double aZoom = 1;
      static Video_Win * aW = Video_Win::PtrWStd(round_ni(Pt2dr(mSz)*aZoom),true,Pt2dr(aZoom,aZoom));
      Fonc_Num f = mImIncid.in_proj();
      ELISE_COPY
      (
             mImIncid.all_pts(),
             Virgule(f,f,f),
             aW->orgb()
      );
      ELISE_COPY
      (
             mImIncid.all_pts(),
             nflag_close_sym(flag_front4(f<0.7*pAramDynAngul)),
             aW->out_graph(Line_St(aW->pdisc()(P8COL::red)))
      );


      aW->clik_in();
   }
   std::cout  << "SssSSzzzzz :: " << mSz << "\n"; 
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
