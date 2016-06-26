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


/**********************************************************************/
/*                                                                    */
/*                            cPMulTiepRed                            */
/*                                                                    */
/**********************************************************************/

cPMulTiepRed::cPMulTiepRed(tMerge * aPM,cAppliTiepRed & anAppli)  :
    mMerge      (aPM),
    mRemoved    (false),
    mSelected   (false),
    mNbCam0     (aPM->NbSom()),
    mNbCamCur   (aPM->NbSom()),
    mVConserved (aPM->VecInd().size(),1)
{
    if (anAppli.ModeIm())
    {
       mP = anAppli.CamMaster().Hom2Cam(aPM->GetVal(0)); 
       mZ = 0.0;  // Faut bien remplir les trou ?
       mPrec = 1.0;


       double aSomRes = 0;
       double aNbRes = 0;

       const std::vector<Pt2dUi2> &  aVP = aPM->Edges();
       for (int aKCple=0 ; aKCple<int(aVP.size()) ; aKCple++)
       {
           int aKC1 = aVP[aKCple].x;
           int aKC2 = aVP[aKCple].y;

           cCameraTiepRed * aCam1 = anAppli.KthCam(aKC1);
           cCameraTiepRed * aCam2 = anAppli.KthCam(aKC2);

           if (aCam1->NameIm() > aCam2->NameIm())
           {
               ElSwap( aKC1, aKC2);
               ElSwap(aCam1,aCam2);
           }

           cLnk2ImTiepRed *  aLnk = anAppli.LnkOfCams(aCam1,aCam2);
           double aRes = anAppli.DefResidual();
           if (aLnk->HasOriRel())
           {
               Pt2dr aP1 =  aCam1->Hom2Cam(aPM->GetVal(aKC1));
               Pt2dr aP2 =  aCam2->Hom2Cam(aPM->GetVal(aKC2));

               CamStenope & aCS1 =  aLnk->CsRel1();
               CamStenope & aCS2 =  aLnk->CsRel2();
               Pt3dr  aPTer = aCS1.PseudoInter(aP1,aCS2,aP2);
               aRes =  (euclid(aP1,aCS1.R3toF2(aPTer)) + euclid(aP2,aCS2.R3toF2(aPTer))) / 2.0;
           }

           aSomRes += aRes;
           aNbRes ++;
       }

       mPrec = aSomRes / aNbRes;

    }
    else
    {
        std::vector<ElSeg3D> aVSeg;
        std::vector<Pt2dr>   aVPt;

        const std::vector<U_INT2>  &  aVecInd = aPM->VecInd() ;
        const std::vector<Pt2df> & aVHom   = aPM-> VecV()  ;

        for (int aKP=0 ; aKP<int(aVecInd.size()) ; aKP++)
        {
             cCameraTiepRed * aCam = anAppli.KthCam(aVecInd[aKP]);
             Pt2dr aPCam = aCam->Hom2Cam(aVHom[aKP]);
             aVPt.push_back(aPCam);
             aVSeg.push_back(aCam->CsOr().Capteur2RayTer(aPCam));
        }

        bool Ok;
        Pt3dr aPTer = InterSeg(aVSeg,Ok);
        double aSomDist = 0.0;
        for (int aKP=0 ; aKP<int(aVecInd.size()) ; aKP++)
        {
             cCameraTiepRed * aCam = anAppli.KthCam(aVecInd[aKP]);
             Pt2dr aPProj = aCam->CsOr().Ter2Capteur(aPTer);
             double aDist = euclid(aPProj,aVPt[aKP]);
             aSomDist += aDist;
        }

        mP = Pt2dr(aPTer.x,aPTer.y);
        mZ = aPTer.z;
        mPrec = aSomDist / (aVecInd.size() -1);
     }
    // std::cout << "PREC " << mPrec << " " << aVecInd.size() << "\n";

     // mGain =   aPM->NbArc()  +  mPrec/1000.0;
}

void  cPMulTiepRed::InitGain(cAppliTiepRed & anAppli)
{
    mGain =  mMerge->NbArc() * (1.0 /(1.0 + ElSquare(mPrec/(anAppli.ThresholdPrecMult() * anAppli.StdPrec()))));
    mGain *= (0.5+ mNbCamCur / double(mNbCam0));
}

bool cPMulTiepRed::Removed() const
{
   return mRemoved;
}

bool cPMulTiepRed::Removable() const
{
   return (mNbCamCur==0);
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
    const std::vector<U_INT2>  & aVNew =  aPNew->mMerge->VecInd() ;
    std::vector<int>  &  aBuf = anAppli.BufICam();
    for (int aK=0 ; aK<int(aVNew.size()) ;aK++)
    {
        aBuf[aVNew[aK]] = 1;
    }

    const std::vector<U_INT2>  & aVCur =  mMerge->VecInd() ;

    for (int aK=0 ; aK<int(aVCur.size()) ; aK++)
    {
         int aKCam = aVCur[aK];
         if (mVConserved[aK]  && (aBuf[aKCam]==1))
         {
             mVConserved[aK] = 0;
             mNbCamCur--;
         }
    }

    InitGain(anAppli);

   // Free Mark index in aBuf
    for (int aK=0 ; aK<int(aVNew.size()) ;aK++)
        aBuf[aVNew[aK]] = 0;
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
