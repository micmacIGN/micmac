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

#include "TiepRed.h"


/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/


cAppliTiepRed::cAppliTiepRed(int argc,char **argv) 
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Name Image 1",  eSAM_IsPatFile),
         LArgMain()  << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
   );

   mEASF.Init(mPatImage);

   if (EAMIsInit(&mCalib))
   {
      StdCorrecNameOrient(mCalib,mEASF.mDir);
   }

   const std::vector<std::string> * aFilesIm = mEASF.SetIm();
   mSetFiles = new std::set<std::string>(aFilesIm->begin(),aFilesIm->end());
   std::cout << " Get Nb Images " <<  aFilesIm->size() << "\n";


   mNM = cVirtInterf_NewO_NameManager::StdAlloc(mEASF.mDir,mCalib);

   for (int aKI = 0 ; aKI<int(aFilesIm->size()) ; aKI++)
   {
       const std::string & aNameIm = (*aFilesIm)[aKI];
       CamStenope * aCS = mNM->OutPutCamera(aNameIm);
       cCameraTiepRed * aCam = new cCameraTiepRed(*this,aNameIm,aCS);
       
       mVecCam.push_back(aCam);
       mMapCam[aNameIm] = aCam;
   }
}



void cAppliTiepRed::Test()
{
   for (int aKI = 0 ; aKI<int(mVecCam.size()) ; aKI++)
   {
       cCameraTiepRed & aCam1 = *(mVecCam[aKI]);
       const std::string & anI1 = aCam1.NameIm();
       // Get list of images sharin tie-P with anI1
       std::list<std::string>  aLI2 = mNM->ListeImOrientedWith(anI1);
       for (std::list<std::string>::const_iterator itL= aLI2.begin(); itL!=aLI2.end() ; itL++)
       {
            const std::string & anI2 = *itL;
            // Test if the file anI2 is in the current pattern
            // As the martini result may containi much more file 
            if (mSetFiles->find(anI2) != mSetFiles->end())
            {
               // The result being symetric, the convention is that some data are stored only for  I1 < I2
               if (anI1 < anI2)
               {
                   cCameraTiepRed & aCam2 = *(mMapCam[anI2]);
                   std::vector<Pt2df> aVP1,aVP2;
                   mNM->LoadHomFloats(anI1,anI2,&aVP1,&aVP2);  // would have worked for I2 > I1 
                   // cXml_Ori2Im aX2 = mNM->GetOri2Im(anI1,anI2); // works only for I1 < I2

                   for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
                   {
                       double aD;
                       Pt3dr aPTer = aCam1.BundleInter(aVP1[aKP],aCam2,aVP2[aKP],aD);
                       std::cout << "AAAAAAAAAAAAAAAaa " << aD << "\n";
                   }
                   std::cout << "NNNN " << anI1 << " " << anI2 << "\n";
                   getchar();

               }
            }
       }
   }

}



int TestOscarTieP_main(int argc,char **argv) 
{
    cAppliTiepRed anAppli(argc,argv);
    anAppli.Test();

    return EXIT_SUCCESS;
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
