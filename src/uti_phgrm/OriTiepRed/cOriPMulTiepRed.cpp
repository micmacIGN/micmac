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

#include "OriTiepRed.h"

NS_OriTiePRed_BEGIN

static bool BugRes=false;

/**********************************************************************/
/*                                                                    */
/*                            cPMulTiepRed                            */
/*                                                                    */
/**********************************************************************/
bool  MergeHasPrec(tMerge * aMerge)
{
    const std::vector<cCMT_U_INT1> &  aVA = aMerge->ValArc();

    for (int aKa=0 ; aKa<int(aVA.size()) ; aKa++)
    {
        if (aVA[aKa].mVal == ORR_MergePrec)
           return true;
    }
    return false;
}

double  cPMulTiepRed::Residual(int aKC1,int aKC2,double aDef,cAppliTiepRed & anAppli) const
{
  cCameraTiepRed * aCam1 = anAppli.KthCam(aKC1);
  cCameraTiepRed * aCam2 = anAppli.KthCam(aKC2);
  if (aCam1->NameIm() > aCam2->NameIm())
  {
     ElSwap( aKC1, aKC2);
     ElSwap(aCam1,aCam2);
  }
  cLnk2ImTiepRed *  aLnk = anAppli.LnkOfCams(aCam1,aCam2,true);
  if ((aLnk==0) || (! aLnk->HasOriRel()))
     return aDef;


  Pt2dr aP1 =  aCam1->Hom2Cam(mMerge->GetVal(aKC1));
  Pt2dr aP2 =  aCam2->Hom2Cam(mMerge->GetVal(aKC2));

  CamStenope & aCS1 =  aLnk->CsRel1();
  CamStenope & aCS2 =  aLnk->CsRel2();
  Pt3dr  aPTer = aCS1.PseudoInter(aP1,aCS2,aP2);
  if ((! aCS1.PIsVisibleInImage(aPTer)) ||  (!aCS2.PIsVisibleInImage(aPTer)))
     return 1e10;
  if (BugRes)
  {
     std::cout << "Vis1="<< aCS1.PIsVisibleInImage(aPTer) << " Vis2=" << aCS2.PIsVisibleInImage(aPTer) << "\n";
     std::cout << "P1="<< aP1 << " P2=" << aP2 << " Ter=" << aPTer << " RProj1=" << aCS1.R3toF2(aPTer) << " RProj2=" << aCS2.R3toF2(aPTer) << "\n";
  }
  return  (euclid(aP1,aCS1.R3toF2(aPTer)) + euclid(aP2,aCS2.R3toF2(aPTer))) / 2.0;
}

void cPMulTiepRed::CompleteArc(cAppliTiepRed & anAppli)
{
    const std::vector<cPairIntType<Pt2df> >  & aVInd = mMerge->VecIT() ;
    if (aVInd.size() == 2) return;

    const std::vector<Pt2di> &  aVP = mMerge->Edges();
    std::vector<int>  &  aBufCpt = anAppli.BufICam();
    std::vector<int>  &  aBufSucc = anAppli.BufICam2();
 
    // int aNbIn = aVP.size();
    // Calcul compteur et successeur

    for (int aKCple=0 ; aKCple<int(aVP.size()) ; aKCple++)
    {
           int aKC1 = aVP[aKCple].x;
           int aKC2 = aVP[aKCple].y;
           aBufCpt[aKC1] ++;
           aBufCpt[aKC2] ++;
           aBufSucc[aKC1] = aKC2;
           aBufSucc[aKC2] = aKC1;
           // std::cout << "KccccCcc  " << aKC1 << " " << aKC2 << "\n";
           // ELISE_ASSERT(aKC1>aKC2,"Internal coherence in complete Arc");
    }



    double  aMoyRes =  MoyResidual(anAppli) ;
    std::vector<int> aVSingl;
    for (int aKSom=0 ; aKSom<int(aVInd.size()) ; aKSom++)
    {
        int anInd1 = aVInd[aKSom].mNum;
        if (aBufCpt[anInd1]==1) 
        {
            double aResMin = 1e2;
            int anIndMin = -1;
            int anInd2 = aBufSucc[anInd1];
            for (int aKSucc=0 ; aKSucc<int(aVInd.size()) ; aKSucc++)
            {
                int anInd3 = aVInd[aKSucc].mNum;
                if (anInd3 != anInd2)
                {
                    double aRes = Residual(anInd1,anInd3,1e3,anAppli);
                    if (aRes < aResMin)
                    {
                        aResMin = aRes;
                        anIndMin = anInd3;
                    }
                }
            }
            // std::cout << "RESmin " << aResMin  << " " << aMoyRes << "\n";
            if ((anIndMin >=0) && (aResMin < 1+2*aMoyRes))
            {
                //set_min_max(anInd1,anIndMin);
                // Important car suppose que les noms soit ordonnes
                if (anAppli.KthCam(anInd1)->NameIm() > anAppli.KthCam(anIndMin)->NameIm())
                {
                   ElSwap(anInd1,anIndMin);
                }
                aBufCpt[anInd1] ++;
                aBufCpt[anIndMin] ++;

                mMerge->NC_Edges().push_back(Pt2di(anInd1,anIndMin));
                mMerge->NC_ValArc().push_back(ORR_MergeCompl);
            }
        }
    }

    for (int aKCple=0 ; aKCple<int(aVP.size()) ; aKCple++)
    {
           int aKC1 = aVP[aKCple].x;
           int aKC2 = aVP[aKCple].y;
           aBufCpt[aKC1] = 0;
           aBufCpt[aKC2] = 0;
           aBufSucc[aKC1] = 0;
           aBufSucc[aKC2] = 0;
    }
    // std::cout << "Coompleete " <<  aNbIn << " " <<  aVP.size() << "\n";
    //getchar();
}

double  cPMulTiepRed::MoyResidual(cAppliTiepRed & anAppli) const
{
   double aSomRes = 0;
   double aNbRes = 0;

   const std::vector<Pt2di> &  aVP = mMerge->Edges();
   for (int aKCple=0 ; aKCple<int(aVP.size()) ; aKCple++)
   {
       double aRes = Residual(aVP[aKCple].x,aVP[aKCple].y,anAppli.DefResidual(),anAppli);

       aSomRes += aRes;
       aNbRes ++;
   }
   if (BugRes)
   {
      std::cout << "cPMulTiepRed::MoyResidual Som=" << aSomRes << " Nb=" << aNbRes << "\n";
   }
   return aSomRes / aNbRes;
}


cPMulTiepRed::cPMulTiepRed(tMerge * aPM,cAppliTiepRed & anAppli)  :
    mMerge      (aPM),
    mRemoved    (false),
    mSelected   (false),
    mNbCam0     (aPM->NbSom()),
    mNbCamCur   (aPM->NbSom()),
    mVConserved (aPM->VecIT().size(),1),
    mHasPrec    (MergeHasPrec(aPM))
{
    if (anAppli.ModeIm())
    {
       // static int aCpt=0 ; aCpt++;
       // BugRes = (aCpt==1032);
       mP = anAppli.CamMaster().Hom2Cam(aPM->GetVal(0)); 
       mZ = 0.0;  // Faut bien remplir les trou ?
       mPrec = MoyResidual(anAppli);
       if (BadNumber(mPrec))
       {
           std::cout << "MASTER IM=" << anAppli.CamMaster().NameIm() << "\n" ; // << " Cpt=" << aCpt << "\n";
           const std::vector<cPairIntType<Pt2df> >  &  aVecIT = aPM->VecIT() ;
           for (int aKP=0 ; aKP<int(aVecIT.size()) ; aKP++)
           {
                cCameraTiepRed * aCam = anAppli.KthCam(aVecIT[aKP].mNum);
                std::cout << "  SEC=" << aCam->NameIm() << "\n";
           }
           ELISE_ASSERT(false,"Bad residual in cPMulTiepRed::cPMulTiepRed");
           // std::cout << "PREC " << mPrec << " In Mode Im "  << "\n";
       }
       if (anAppli.DoCompleteArc())
           CompleteArc(anAppli);
    }
    else
    {
        std::vector<ElSeg3D> aVSeg;
        std::vector<Pt2dr>   aVPt;

        const std::vector<cPairIntType<Pt2df> >  &  aVecIT = aPM->VecIT() ;
        // const std::vector<Pt2df> & aVHom   = aPM-> VecV()  ;

        for (int aKP=0 ; aKP<int(aVecIT.size()) ; aKP++)
        {
             cCameraTiepRed * aCam = anAppli.KthCam(aVecIT[aKP].mNum);
             Pt2dr aPCam = aCam->Hom2Cam(aVecIT[aKP].mVal);
             aVPt.push_back(aPCam);
             aVSeg.push_back(aCam->CsOr().Capteur2RayTer(aPCam));
        }

        bool Ok;
        Pt3dr aPTer = InterSeg(aVSeg,Ok);
        double aSomDist = 0.0;
        for (int aKP=0 ; aKP<int(aVecIT.size()) ; aKP++)
        {
             cCameraTiepRed * aCam = anAppli.KthCam(aVecIT[aKP].mNum);
             Pt2dr aPProj = aCam->CsOr().Ter2Capteur(aPTer);
             double aDist = euclid(aPProj,aVPt[aKP]);
             aSomDist += aDist;
        }

        mP = Pt2dr(aPTer.x,aPTer.y);
        mZ = aPTer.z;
        mPrec = aSomDist / (aVecIT.size() -1);
     }
    // std::cout << "PREC " << mPrec << " " << aVecInd.size() << "\n";

     // mGain =   aPM->NbArc()  +  mPrec/1000.0;

}

void  cPMulTiepRed::InitGain(cAppliTiepRed & anAppli)
{
    mGain =  mMerge->Edges().size() * (1.0 /(1.0 + ElSquare(mPrec/(anAppli.ThresholdPrecMult() * anAppli.StdPrec()))));
    mGain *= (0.5+ mNbCamCur / double(mNbCam0));

    if (mHasPrec)
    {
       mGain += 1000;
    }
}

bool cPMulTiepRed::Removed() const
{
   return mRemoved;
}

bool cPMulTiepRed::Removable() const
{
   return (mNbCamCur==0) && (!mHasPrec);
}


void cPMulTiepRed::Remove()
{
    mRemoved = true;
}

bool cPMulTiepRed::Selected() const
{
   return mSelected;
}

void cPMulTiepRed::SetSelected()
{
    mSelected = true;
}


void cPMulTiepRed::UpdateNewSel(const cPMulTiepRed * aPNew,cAppliTiepRed & anAppli)
{
   // Mark index of aPNew as existing in buf
    const std::vector<cPairIntType<Pt2df> >  & aVNew =  aPNew->mMerge->VecIT() ;
    std::vector<int>  &  aBuf = anAppli.BufICam();
    for (int aK=0 ; aK<int(aVNew.size()) ;aK++)
    {
        aBuf[aVNew[aK].mNum] = 1;
    }

    const std::vector<cPairIntType<Pt2df> >  & aVCur =  mMerge->VecIT() ;

    for (int aK=0 ; aK<int(aVCur.size()) ; aK++)
    {
         int aKCam = aVCur[aK].mNum;
         if (mVConserved[aK]  && (aBuf[aKCam]==1))
         {
             mVConserved[aK] = 0;
             mNbCamCur--;
         }
    }

    InitGain(anAppli);

   // Free Mark index in aBuf
    for (int aK=0 ; aK<int(aVNew.size()) ;aK++)
        aBuf[aVNew[aK].mNum] = 0;
}

void cPMulTiepRed::SetDistVonGruber(const double & aDist,const cAppliTiepRed & anAppli)
{
    mDMin = aDist;
    mGain = mDMin / (1+2.0*mPrec/anAppli.StdPrec());
}

void cPMulTiepRed::ModifDistVonGruber(const double & aDist,const cAppliTiepRed & anAppli)
{
    SetDistVonGruber(ElMin(aDist,mDMin),anAppli);
}

bool cPMulTiepRed::HasPrec() const
{
   return mHasPrec;
}


NS_OriTiePRed_END






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
