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
#include "TpPPMD.h"


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

cTD_Camera::cTD_Camera(const std::string & aName) :
    mName (aName),
    mCIO  (StdGetFromPCP(aName,CalibrationInternConique)),
    mCS (CamOrientGenFromFile(NameWithoutDir(aName),cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aName))))
{
}

Pt2dr cTD_Camera::Ter2Image(const Pt3dr & aPTer) const
{
    return mCS->R3toF2(aPTer);
}

std::vector<cTD_Camera> cTD_Camera::RelvtEspace
                        (
                                    const Pt3dr & aPTer1, const Pt2dr & aPIm1,
                                    const Pt3dr & aPTer2, const Pt2dr & aPIm2,
                                    const Pt3dr & aPTer3, const Pt2dr & aPIm3
                        )
{
    std::list<ElRotation3D> aLR;


    mCS->OrientFromPtsAppui(aLR,aPTer1,aPTer2,aPTer3,aPIm1,aPIm2,aPIm3);

    std::vector<cTD_Camera>  aRes;

    for (std::list<ElRotation3D>::const_iterator itR=aLR.begin(); itR!=aLR.end(); itR++)
    {
          cTD_Camera aNew = *this;
          aNew.mCS = mCS->Dupl();
          aNew.mCS->SetOrientation(*itR);
          aRes.push_back(aNew);
    }


    return aRes;
}

void cTD_Camera::Save(const std::string & aName) const
{
    MakeFileXML
    (
       mCS->StdExportCalibGlob(),
       aName
    );
}

double cTD_Camera::Focale () const
{
    return mCIO.F();
}

double  cTD_Camera::R3 () const
{
    return mCIO.CalibDistortion()[0].ModRad().Val().CoeffDist()[0];
}

Pt2dr    cTD_Camera::SzCam() const 
{
   return Pt2dr(mCS->Sz());
}


cTD_Camera cTD_Camera::NewCam(double aFoc , double aR3)
{
     cTD_Camera aRes = *this;

     cOrientationConique aCO = mCS->StdExportCalibGlob();
     aCO.Interne().Val().F() = aFoc;
     std::vector<double> & aVC = aCO.Interne().Val().CalibDistortion()[0].ModRad().Val().CoeffDist();
     if (aVC.size() ==0)
        aVC.push_back(0);
     aVC[0] = aR3;

     aRes.mCIO = aCO.Interne().Val();
     aRes.mCS = Cam_Gen_From_XML(aCO,cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(mName)),mCS->IdentCam())->CS();

     return aRes;
}


Pt2dr  cTD_Camera::Ter2Im(const Pt3dr & aPTer) const
{
   return mCS->R3toF2(aPTer);
}

Pt3dr cTD_Camera::ImAndProf2Ter(const Pt2dr & aPTer,double aProf) const
{
   return mCS->ImEtProf2Terrain(aPTer,aProf);
}

double  cTD_Camera::ProfMoy() const
{
   return mCS->GetProfondeur();
}

double cTD_Camera::StepProfOnePixel(const cTD_Camera & aCam2) const
{
    Pt3dr aC1 = mCS->VraiOpticalCenter();
    Pt3dr aC2 = aCam2.mCS->VraiOpticalCenter();

    double aBase = euclid(aC1-aC2);
    double aProf = (ProfMoy() + aCam2.ProfMoy()) / 2.0;

    double aF = (Focale() + aCam2.Focale()) / 2.0;


    return ElSquare(aProf) / ( aF * aBase);


}






/********************************************************************/
/*                                                                  */
/*         cTD_Prof                                                 */
/*                                                                  */
/********************************************************************/



void GenereAppar32(CamStenope *aCS,const std::string & aName ,int aNb,double aNoiseGauss,double aProbaBigNoise)
{
    Pt2dr aSz = Pt2dr(aCS->Sz());
    Box2dr aBox(Pt2dr(0,0),aSz);

    cListeAppuis1Im aL;

    for (int aK=0 ; aK< aNb ; aK++)
    {
        Pt2dr aPIm =  aBox.RandomlyGenereInside();
        double aProf = 3.0 * (1+NRrandom3());
        Pt3dr aPTer = aCS->ImEtProf2Terrain(aPIm,aProf);
        aPIm =  aPIm + Pt2dr(NRrandC(),NRrandC()) * aNoiseGauss;
        if (NRrandom3()<aProbaBigNoise)
           aPIm =  aBox.RandomlyGenereInside();

        cMesureAppuis aM;
        aM.Num().SetVal(aK);
        aM.Im() = aPIm;
        aM.Ter() = aPTer;
        aL.Mesures().push_back(aM);
    }
    // aL.NameImage().SetVal(aCam.mName);
    MakeFileXML(aL,aName+"_PtsTest.xml");
}

int TD_GenereAppuis_main(int argc,char ** argv)
{
    std::string aNameCam,toto;
    int aNbPts;
    double aWhiteNoise,aProbaOut;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameCam,"Name of camera")
                    << EAMC(aNbPts,"Nuber of point")
                    << EAMC(aWhiteNoise,"White noise added")
                    << EAMC(aProbaOut,"Proba of Out Layers"),
        LArgMain()  << EAM(toto,"toto",true,"Do no stuff")
    );

    // cTD_Camera aCam(aNameCam);
    CamStenope * aCS =  CamOrientGenFromFile
                        (
                              NameWithoutDir(aNameCam),
                              cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aNameCam))
                        );


    GenereAppar32(aCS,aNameCam,aNbPts,aWhiteNoise,aProbaOut);

    return 0;
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
