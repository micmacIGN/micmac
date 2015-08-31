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
#include "Apero.h"



cInterpolateurIm2D<U_INT2>* cOneVisuPMul::Interp() const { return mInterp; }
const Pt3dr&  cOneVisuPMul::PTer00()   const { return mPTer00; }
const Pt3dr&  cOneVisuPMul::X_VecTer() const { return mX_VecTer; }
const Pt3dr&  cOneVisuPMul::Y_VecTer() const { return mY_VecTer; }


#if (ELISE_NO_VIDEO)


void cAppliApero::DoShowPtsMult(const std::list<cVisuPtsMult> & aLVPM)
{
    ELISE_ASSERT(aLVPM.size()==0,"cAppliApero::DoShowPtsMult / ELISE_NO_VIDEO");
}
#else

/*****************************************************/
/*                                                   */
/*         cAppliApero                               */
/*                                                   */
/*****************************************************/

cAuxVisuPMul::cAuxVisuPMul
(
    const cVisuPtsMult &  aVPM,
    Video_Win *           aWRef,
    Video_Win::ePosRel    aPos
) :
    mSz (aVPM.SzWAux().Val()),
    mW (*aWRef,aPos,Pt2di(mSz,mSz))
{
   clear();
}

void cAuxVisuPMul::clear()
{
   ELISE_COPY
   (
       mW.all_pts(),
       P8COL::red,
       mW.odisc()
   );
}


Video_Win * cAuxVisuPMul::W()
{
   return & mW;
}


void cAuxVisuPMul::SetChc(Pt2dr aTr,double aSc)
{
   mW = mW.chc(aTr,Pt2dr(aSc,aSc));
}

/*****************************************************/
/*                                                   */
/*         cOneVisuPMul                              */
/*                                                   */
/*****************************************************/

class cOVP_Inter : public EliseStdImageInteractor
{
   public :
     cOVP_Inter
     (
             cOneVisuPMul & anOVP,
             Video_Win aW,
             ElPyramScroller & aPyr
     )  :
          EliseStdImageInteractor (aW,aPyr,2,5,4),
          mOVP (anOVP)
     {
     }

     cOneVisuPMul & mOVP;

     void OnEndTranslate(Clik){mOVP.ShowVect();}
     void OnEndScale(Clik aCl)
     {
          EliseStdImageInteractor::OnEndScale(aCl);
          mOVP.ShowVect();
     }

};

cOneVisuPMul::cOneVisuPMul
(    
     const cVisuPtsMult & aVPM,
     cAppliApero &        anAppli
)  :
   mAppli   (anAppli),
   mVMP     (aVPM),
   mModeVisu(true),
   mCam     (mAppli.PoseFromName(aVPM.Cam1())),
   mNameCam (mCam->Name()),
   mOLM     (mAppli.PackMulOfIndAndNale(mVMP.Id(),mNameCam)),
   mNameF   (mAppli.DC()+mNameCam),
   mFileIm  (Tiff_Im::UnivConvStd(mNameF)),
   mSzF     (mFileIm.sz()),
   mRatio   (mVMP.SzWPrinc().Val()/double(ElMax(mSzF.x,mSzF.y))),
   mSzW     (round_ni(Pt2dr(mSzF)*mRatio)),
   mWPrinc  (mModeVisu ? (Video_Win::PtrWStd(mSzW)) : 0),
   mVVE     (mModeVisu ? (new VideoWin_Visu_ElImScr(*mWPrinc,mFileIm)) : 0),
   mPyr     (mModeVisu ? (ElImScroller::StdPyramide(*mVVE,mNameF)) : 0),
   // i mESII    (new EliseStdImageInteractor(*mWPrinc,*mPyr,2,5,4))
   mESII    (mModeVisu ? (new  cOVP_Inter(*this,*mWPrinc,*mPyr)) : 0),
   mInterp  (new cInterpolBilineaire<U_INT2>)
{

    cAuxVisuPMul * aFirstLinePrec = 0;
    for (int aKx=0 ; aKx< aVPM.NbWAux().x ; aKx++)
    {
       for (int aKy=0 ; aKy< aVPM.NbWAux().y ; aKy++)
       {
            Video_Win * aWRef=0;
            Video_Win::ePosRel aPos;
            if (aKy>0)
            {
               aWRef =  mVWaux.back()->W();
               aPos = Video_Win::eBasG;
            }
            else if (aKx>0)
            {
               aWRef = aFirstLinePrec->W();
               aPos = Video_Win::eDroiteH;
            }
            else
            {
               aWRef = mWPrinc;
               aPos = aVPM.AuxEnDessous() ? Video_Win::eBasG : Video_Win::eDroiteH ;
            }
            mVWaux.push_back(new cAuxVisuPMul(aVPM,aWRef,aPos));
            if (aKy==0)
               aFirstLinePrec = mVWaux.back();
       }
    }
}

void cOneVisuPMul::ShowVect()
{
   const std::vector<cOnePtsMult *> aVMP = mOLM->VPMul();
   for (int aK=0 ; aK<int(aVMP.size()) ; aK++)
   {
        int aNb = aVMP[aK]->NPts().NbPts();
        int aCoul = P8COL::green;
        if (aNb==3) aCoul = P8COL::blue;
        if (aNb> 3) aCoul = P8COL::yellow;
        ShowCenterOnePMul(*aVMP[aK],aCoul);
   }
}

void cOneVisuPMul::ShowCenterOnePMul(cOnePtsMult & aPM,int aCol)
{
     mWPrinc->draw_circle_abs(mESII->U2W(aPM.P0()),1.5,mWPrinc->pdisc()(aCol));

}

void cOneVisuPMul::InterAction()
{
    while (1)
    {
        Clik aCl = mESII->clik_press();
        cOnePtsMult & aPM = mOLM->PMulLPP(mESII->W2U(aCl._pt));
        ShowCenterOnePMul(aPM,P8COL::red);
        std::cout << "NB Mult " << aPM.NPts().NbPts() << "\n";

        DoOnePMul(aPM);
    }

}

void cOneVisuPMul::DoOnePMul(cOnePtsMult & aPM )
{
   std::vector<double> aVPds;


   int aNbOk=   aPM.InitPdsPMul(1.0,aVPds);
   if (aNbOk <2) 
       return;
   mPTer00 = aPM.OCM()->LiaisTer()->CalcPTerIFC_Robuste
                 (
                      0.5,
                      aPM.NPts(),
                      aVPds
                 );


    const CamStenope * aCS0 = aPM.GenPose0()->DownCastPoseCamNN()->CurCam();

    // cInterpolBilineaire<U_INT2> anIB;
    // cRecorrel aRC(anIB,aPM.Pose0()


    Pt2dr aPIm = aCS0->R3toF2(mPTer00);

    Pt2dr aDirBase(1,1);
    Pt2dr  aDirX = vunit(aDirBase);
    Pt2dr  aDirY = aDirX*Pt2dr(0,1);

    mX_VecTer = aCS0->F2AndZtoR3(aPIm+aDirX,mPTer00.z) -mPTer00;
    mY_VecTer = aCS0->F2AndZtoR3(aPIm+aDirY,mPTer00.z) -mPTer00;
    
   
   // aVPter.push_back(aRes->mPTer);
   // std::cout << aVPter[0] << aVPter[1] << euclid(aVPter[0]-aVPter[1]) << "\n";

   


   if (mModeVisu)
   {
       std::vector<double> aVPds;
       const cResiduP3Inc * aRes = aPM.ComputeInter(1.0,aVPds);
       if ( aRes) 
       {
          for (int aKPose=0 ; aKPose<int(aRes->mEcIm.size()) ; aKPose++)
          {
              std::cout << euclid(aRes->mEcIm[aKPose]) << "\n";
          }
       }
  }

   std::list<cOnePtsMult *> aL;
   aL.push_back(&aPM);
      // On charge bcp + que necessaire
   int aSzA = mVMP.SzWAux().Val();
   mAppli.LoadImageForPtsMul(Pt2di(0,0),Pt2di(aSzA,aSzA),aL);

   std::vector<cPoseCam*>  aVPCL = mAppli.VecLoadedPose(aPM,6);

   if (int(aVPCL.size()) < 2) 
      return;

   if (mModeVisu)
   {
      for (int aKW=0; aKW<int(mVWaux.size()) ; aKW++)
      {
          mVWaux[aKW]->clear();
      }
   }

   int aSzV0 = 5;
   double aStep0 = 1.0;
    // const CamStenope * aCS0 = aPM.GenPose0()->DownCastPoseCamNN()->CurCam();
   cRecorrel aRC0(*this,aPM.GenPose0()->DownCastPoseCamNN(),aSzV0,aStep0);


   double                    aDistPdsErr = 0.25;
   std::vector<cBasicGeomCap3D *> aVCS;
   cNupletPtsHomologues      aNewNuple(0,1.0);
   std::vector<double>       aVPdsNew;
   std::vector<double>       aVCorrel;

   bool OKAllSift = true;
   for (int aKPose=0 ; aKPose<int(aVPCL.size()) ; aKPose++)
   {
       Box2di  aBox = aVPCL[aKPose]->BoxIm();
       Pt2dr aDec = Pt2dr(aBox._p0);
       Im2D_U_INT2 anIm = aVPCL[aKPose]->Im();

       const CamStenope * aCS = aVPCL[aKPose]->CurCam();
       Pt2dr aPImAbs = aCS->R3toF2(mPTer00);
       Pt2dr aPImLoc = aPImAbs - aDec;

       cRecorrel aRCK(*this,aVPCL[aKPose],aSzV0,aStep0);
       int anInd = aPM.IndOfPose(aVPCL[aKPose]);

       if (mModeVisu) 
       {
           std::cout << aVPCL[aKPose]->Name() 
                     <<  " " << anInd
                     <<  " " << aRCK.OneCorrelOfDec(Pt2dr(0,0),aRC0)
                     << "\n";
       }

ElTimer aChr;
             
       aRCK.ExploreVois(10,1.0,aRC0);
       aRCK.ExploreVois(3,0.5,aRC0);
       aRCK.DescAs_Optim2DParam(aRC0);


       double aCorrel = aRCK.BestCorrel();
       double aDistInit = euclid(aRCK.BestPImAbs()-aPImAbs);
       bool OkPts =    (aCorrel>mVMP.SeuilCorrel().Val())
                    && (aDistInit<mVMP.MaxDistProjCorr().Val());
 
       if ((anInd>=0) && (!OkPts)) 
          OKAllSift = false;


       if (mModeVisu) 
       {
             std::cout  <<"COR-OPT="<<aCorrel
                        <<" D="<<aDistInit
                        <<"T="<< aChr.uval()<<"\n";
       }

       aVCorrel.push_back(aRCK.BestCorrel());
       Pt2dr aPOptImLoc = aRCK.BestPImAbs() - aDec;

       if (mModeVisu) 
       {
           int aZ = mVMP.ZoomWAux().Val();
           int aSz = mVMP.SzWAux().Val() / aZ;

           if (aKPose<int(mVWaux.size()))
           {
                Pt2di aDecVisu = round_ni(aPImLoc-Pt2dr(aSz,aSz)/2.0);
                mVWaux[aKPose]->SetChc(Pt2dr(aDecVisu),aZ);
                Video_Win * aW =  mVWaux[aKPose]->W();

                double aS0,aS1,aS2;


                Fonc_Num aFIn = aVPCL[aKPose]->Im().inside();
                Fonc_Num aF = aVPCL[aKPose]->Im().in(0);

                ELISE_COPY
                (
                     aW->all_pts(),
                     Virgule(aFIn,aF,ElSquare(aF)),
                     Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
                );
                aS1 /= aS0;
                aS2 /= aS0;
                aS2 -= ElSquare(aS1);
                aS2 = sqrt(ElMax(1.0,aS2));


                ELISE_COPY
                (
                     aW->all_pts(),
                      255 * erfcc ((aF-aS1)/aS2),
                     aW->ogray()
                );

                if (0)
                {
                     for (int aDx=-aSzV0; aDx<=aSzV0 ;aDx++)
                        for (int aDy=-aSzV0; aDy<=aSzV0 ;aDy++)
                        {
                              Pt3dr aPT =  mPTer00 
                                         + mX_VecTer * double(aDx) 
                                         + mY_VecTer * double(aDy);
                              Pt2dr aPIm = aCS->R3toF2(aPT)-aDec;
                              aW->draw_circle_loc(aPIm,0.3,aW->pdisc()(P8COL::green));
                        }
                }

                aW->draw_circle_loc(aPImLoc,0.5,aW->pdisc()(P8COL::green));
                aW->draw_circle_loc(aPOptImLoc,0.5,aW->pdisc()(P8COL::red));
           }
       }
       aVCS.push_back(aVPCL[aKPose]->NC_CurCam());
       aVPdsNew.push_back(OkPts?1.0:0.0);
       aNewNuple.AddPts(aRCK.BestPImAbs());
   }

   if (!OKAllSift)
      return;

   Pt3dr aNewP = CalcPTerIFC_Robuste(aDistPdsErr,aVCS,aNewNuple,aVPdsNew);


   for (int aKPose=0 ; aKPose<int(aVCS.size()) ; aKPose++)
   {
       bool OkPts = true;
       bool IsSift = false;

       Pt2dr aPImAbs = aVCS[aKPose]->Ter2Capteur(aNewP);

       double aDistProj = euclid(aPImAbs,aNewNuple.PK(aKPose));
       if (aDistProj > mVMP.MaxDistReproj().Val())
          OkPts = false;


       if (mModeVisu)
       {
          std::cout <<  "  D-PROJ == "<< aDistProj 
                    << " COR " << aVCorrel[aKPose] << "\n";
       }
       int anInd =  aPM.IndOfPose(aVPCL[aKPose]);
       if (anInd >=0 )
       {
           Pt2dr aPSift = aPM.PK(anInd);

           double aDIm = euclid(aNewNuple.PK(aKPose),aPSift);
           double aDProj = euclid(aPImAbs,aPSift);
           OkPts =      OkPts 
                    && (aDIm<mVMP.MaxDistSift().Val())
                    && (aDProj<mVMP.MaxDistReproj().Val());

           if (mModeVisu)
           {
              std::cout << "DSIFT :: " 
                        << " S-Im= " << aDIm 
                        << " S-Proj=" << aDProj << "\n";
           }
       }
       if (IsSift && (! OkPts))
          OKAllSift = false;
       if (! OkPts)
          aVPdsNew[aKPose] = 0.0;
   }

   if (mModeVisu)
   {
      for (int aKPose=0 ; aKPose<int(aVCS.size()) ; aKPose++)
      {
          std::cout <<  ((aVPdsNew[aKPose]>0) ? "+ " :"- ")
                    << ((aPM.IndOfPose(aVPCL[aKPose]) >=0) ? " Sift " : " NewP ")
                    << aVPCL[aKPose]->Name() 
                    << "\n";
      }
      std::cout << " ---------------------------------\n";
   }

}




/*****************************************************/
/*                                                   */
/*         cAppliApero                               */
/*                                                   */
/*****************************************************/
 
void cAppliApero::DoShowPtsMult(const std::list<cVisuPtsMult> & aLVPM)
{
   for 
   (
       std::list<cVisuPtsMult>::const_iterator itVPM=aLVPM.begin();
       itVPM!=aLVPM.end();
       itVPM++
   )
   {
      mVisuPM.push_back(new cOneVisuPMul(*itVPM,*this));
   }


   for 
   (
       std::list<cOneVisuPMul *>::const_iterator itVPM=mVisuPM.begin();
       itVPM!=mVisuPM.end();
       itVPM++
   )
   {
       (*itVPM)->InterAction();
   }
}

#endif // (ELISE_NO_VIDEO)



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
