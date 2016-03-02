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
/*                         cCameraTiepRed                             */
/*                                                                    */
/**********************************************************************/

cCameraTiepRed::cCameraTiepRed
(
    cAppliTiepRed &         anAppli,
    const std::string &     aName,
    CamStenope *            aCam
) :
   mAppli  (anAppli),
   mNameIm (aName),
   mCS     (aCam),
   mNbPtsHom2Im  (0),
   mNum          (-1)
{
}

CamStenope  & cCameraTiepRed::CS() {return *mCS;}
const int & cCameraTiepRed::NbPtsHom2Im() const {return mNbPtsHom2Im;}
bool  cCameraTiepRed::SelectOnHom2Im() const
{
    return mNbPtsHom2Im >= mAppli.ThresholdTotalNbPts2Im();
}

void cCameraTiepRed::SetNum(int aNum)
{
   mNum = aNum;
}
const int & cCameraTiepRed::Num() const
{
   return mNum;
}




Pt2dr cCameraTiepRed::Hom2Cam(const Pt2df & aP) const
{
     Pt3dr aQ(aP.x,aP.y,1.0);
     return mCS->L3toF2(aQ);
}


const std::string cCameraTiepRed::NameIm() const { return mNameIm; }

Pt3dr cCameraTiepRed::BundleIntersection(const Pt2df & aPH1,const cCameraTiepRed & aCam2,const Pt2df & aPH2,double & Precision) const
{

     Pt2dr aPC1 = Hom2Cam(aPH1);
     Pt2dr aPC2 = aCam2.Hom2Cam(aPH2);

     ElSeg3D aSeg1 = mCS->Capteur2RayTer(aPC1);
     ElSeg3D aSeg2 = aCam2.mCS->Capteur2RayTer(aPC2);

     bool Ok;
     double aD;
     Pt3dr aRes= InterSeg(aSeg1.P0(),aSeg1.P1(),aSeg2.P0(),aSeg2.P1(),Ok,&aD);

     if (Ok)
     {
         Pt2dr aRP1 = mCS->Ter2Capteur(aRes);
         Pt2dr aRP2 = aCam2.mCS->Ter2Capteur(aRes);
         Precision = (euclid(aRP1-aPC1)+euclid(aRP2-aPC2)) / 2.0;
     }
     else
     {
        Precision = 1e20;
     }
     
     return aRes;
    
}

void cCameraTiepRed::LoadHom(cCameraTiepRed & aCam2)
{


    // Declare Input Tie Points
    std::vector<Pt2df> aVPIn1,aVPIn2;
    // Load Input
    mAppli.NM().LoadHomFloats(NameIm(),aCam2.NameIm(),&aVPIn1,&aVPIn2);  // would have worked for I2 > I1 
    Box2dr aBox = mAppli.ParamBox().Box();
    double aThresh = mAppli.ThresoldPrec2Point();

     // Create a connexion with initialy no tie points
    cLnk2ImTiepRed * aLnk = new cLnk2ImTiepRed(this,&aCam2);
    std::vector<Pt2df> & aVPOut1 = aLnk->VP1();
    std::vector<Pt2df> & aVPOut2 = aLnk->VP2();

    // Filter the ties points that are inside the current tiles and
    // have "good" intersection
    for (int aKP=0 ; aKP<int(aVPIn1.size()) ; aKP++)
    {
        double aD; // store the reprojection error

        Pt3dr aPTer = BundleIntersection(aVPIn1[aKP],aCam2,aVPIn2[aKP],aD);
        if ( (aD< aThresh) && aBox.inside(Pt2dr(aPTer.x,aPTer.y)) )
        {
            aVPOut1.push_back(aVPIn1[aKP]);
            aVPOut2.push_back(aVPIn2[aKP]);
        }
        //                std::cout << "AAAAAAAAAAAAAAAaa " << aD << aPTer << "\n";
    }

    // If enough tie point , memorize the connexion 

    if (int(aVPOut1.size()) >= mAppli.ThresholdNbPts2Im())
    {
        // Update counters
        mNbPtsHom2Im +=  aVPOut1.size();  
        aCam2.mNbPtsHom2Im +=  aVPOut1.size();
        // Ask application to memorize
        mAppli.AddLnk(aLnk);
    }
    else
    {
         delete aLnk;
    }
}
  
void cCameraTiepRed::AddCamBox(cCameraTiepRed* aCam2,int aKBox)
{
   mMapCamBox[aCam2].push_back(aKBox);
}


void cCameraTiepRed::SaveHom(cCameraTiepRed* aCam2,const std::list<int> & aLBox)
{


    ElPackHomologue aRes;
    for (std::list<int>::const_iterator itI=aLBox.begin(); itI!=aLBox.end() ; itI++)
    {
         std::string aName = mAppli.NameHomol(NameIm(),aCam2->NameIm(),*itI);
         if (ELISE_fp::exist_file(aName))
         {
             ElPackHomologue aPack = ElPackHomologue::FromFile(aName);
             aRes.Add(aPack);

         }
    }

    if (aRes.size())
    {
         std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ mAppli.StrOut() + "@dat";
         std::string aNameH = mAppli.ICNM()->Assoc1To2(aKeyH,NameIm(),aCam2->NameIm(),true);
         aRes.StdPutInFile(aNameH);
         // std::string aNameH = mAppli
    }
}

void  cCameraTiepRed::SaveHom()
{
    for (std::map<cCameraTiepRed*,std::list<int> >::const_iterator itM=mMapCamBox.begin(); itM!=mMapCamBox.end() ;itM++)
       SaveHom(itM->first,itM->second);
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
