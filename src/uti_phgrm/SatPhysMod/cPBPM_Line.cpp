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


#include "SatPhysMod.h"

/**********************************************************************/
/*                                                                    */
/*                      cPushB_GeomLine                               */
/*                                                                    */
/**********************************************************************/


cPushB_GeomLine::cPushB_GeomLine(const cPushB_PhysMod * aPBPM,int aNbSampleX,double anY) :
    mPBPM    (aPBPM),
    mNbX     (aNbSampleX),
    mY       (anY),
    mPlanRay (Pt3dr(0,0,0),Pt3dr(1,0,0),Pt3dr(0,1,0))
{

   // Computation of segments and centers
    std::vector<ElSeg3D> aVSeg;
    bool Ok;
    for (int aKS=0 ; aKS<=aNbSampleX ; aKS++)
    {
        aVSeg.push_back(mPBPM->Im2GeoC(PIm(aKS)));
    }
    mCenter = InterSeg(aVSeg,Ok);
    mCUnRot = mCenter;
    double aSomD=0;
    mMaxResInt = 0;

    Pt3dr aSpeed = vunit(aPBPM->Rough_GroundSpeed(anY));


   // Computation of directions
    std::vector<Pt3dr> aVDirOrient;
    {
       std::vector<Pt3dr> aVDirCalcPlan;
       for (int aKS=0 ; aKS<int(aVSeg.size()) ; aKS++)
       {
            double aDist  = aVSeg[aKS].DistDoite(mCenter);
            aSomD += aDist;
            ElSetMax(mMaxResInt,aDist);

            Pt3dr aDir = vunit(aVSeg[aKS].Mil()-mCenter);
            aVDirOrient.push_back(aDir);

            // we want to enforce the fact that the plane contains (0,0,0) so we add P and -P
            // to this observation
            aVDirCalcPlan.push_back(aDir);
            aVDirCalcPlan.push_back(-aDir);
       }
       mPlanRay = cElPlan3D(aVDirCalcPlan,0);
    }
    mResInt = aSomD / aVSeg.size();

    // R=CoordPlan2Euclid()  affine rotation  such that, if P is in plane coordinates
    // R(P) is in global coordinates;   

    ElRotation3D aRe2p = mPlanRay.CoordPlan2Euclid().inv();
    mAxeY = mPlanRay.Norm();
    if (scal(aSpeed,mAxeY) < 0) 
       mAxeY = -mAxeY;

    // Pt3dr aX0 = vunit(mPlanRay.Proj(aRe2p.ImAff(aVDirPlan[aVSeg.size()/2])));
    int aK0 = (int)(aVSeg.size() / 2);
    mAxeZ = DirOnPlan(aVDirOrient[aK0]);
    mAxeX = vunit(mAxeY ^mAxeZ);

    mMoyDistPlan = 0.0;
    mMaxDistPlan = 0.0;
    double aTetaMoy = 0.0;
    for (int aKS=0 ; aKS<int(aVSeg.size()) ; aKS++)
    {
        Pt3dr aD = aVDirOrient[aKS];
        double aDPlan = ElAbs(scal(mAxeY,aD));
        mMoyDistPlan += aDPlan;
        ElSetMax(mMaxDistPlan,aDPlan);

        Pt3dr aProjDir = DirOnPlan(aD);
        double aScalZ = ElMax(-1.0,ElMin(1.0,scal(mAxeZ,aProjDir)));
        double aScalX = ElMax(-1.0,ElMin(1.0,scal(mAxeX,aProjDir)));
        double aTeta = atan2(aScalX,aScalZ);
        aTetaMoy += aTeta;
    }
    aTetaMoy /= aVSeg.size();

    mAxeZ = DirOnPlan(mAxeZ*cos(aTetaMoy) + mAxeX * sin(aTetaMoy));
    mAxeX = vunit(mAxeY ^mAxeZ);


    mMoyDistPlan /= aVSeg.size();
    for (int aKS=0 ; aKS<int(aVSeg.size()) ; aKS++)
    {
        Pt3dr aD = DirOnPlan(aVDirOrient[aKS]);
  
        double aZ = scal(mAxeZ,aD);
        double aX = scal(mAxeX,aD);
        mCalib.push_back(aX/aZ);
    }
}


// compute the projection of P on mPlanRay, then make it a unit vector
Pt3dr cPushB_GeomLine::DirOnPlan(const Pt3dr & aP) const
{
   return vunit(mPlanRay.Proj(aP));
}


ElMatrix<double>  cPushB_GeomLine::MatC2M() const
{
    return MatFromCol(mAxeX,mAxeY,mAxeZ);
}

ElMatrix<double>  cPushB_GeomLine::MatC1ToC2(cPushB_GeomLine & aCam2) const
{
   return aCam2.MatC2M().transpose()   * MatC2M();
}






double  cPushB_GeomLine::XIm(int aKx) const
{
    return (double(mPBPM->SwapXY() ? mPBPM->Sz().y : mPBPM->Sz().x) * (aKx+0.5) )/(mNbX+1);
}


Pt2dr cPushB_GeomLine::PIm(int aKx) const
{
   return  mPBPM->SwapXY() ? Pt2dr(mY,XIm(aKx)) :  Pt2dr(XIm(aKx),mY) ;
}

const Pt3dr &   cPushB_GeomLine::Center() const {return mCenter;}
const double  & cPushB_GeomLine::MoyResiduCenter() const {return mResInt;}
const double  & cPushB_GeomLine::MaxResiduCenter() const {return mMaxResInt;}
const double  & cPushB_GeomLine::MoyDistPlan() const {return mMoyDistPlan;}
const double  & cPushB_GeomLine::MaxDistPlan() const {return mMaxDistPlan;}


const Pt3dr &   cPushB_GeomLine::CUnRot() const {return mCUnRot;}
Pt3dr &   cPushB_GeomLine::CUnRot() {return mCUnRot;}
const std::vector<double> cPushB_GeomLine::Calib() const {return mCalib;}


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
