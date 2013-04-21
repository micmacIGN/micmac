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

#include "Digeo.h"


static Video_Win aWTesD = Video_Win::WStd(Pt2di(100,100),1.0);

template <class Type,class tBase> void Show_Octave(cTplOctDig<Type> * anOct)
{
  const std::vector<cTplImInMem<Type> *> &  aVIm = anOct->cTplOctDig<Type>::VTplIms();


  for (int aK=0 ; aK<int(aVIm.size()) ; aK++)
  {
       cTplImInMem<Type> & anIm = *(aVIm[aK]);
       Im2D<Type,tBase> aTIm = anIm.TIm() ;  // L'image qu'il faut manipuler
       std::cout << "   #  Sz " << aTIm.sz() << " SInit:" <<  anIm.ScaleInit() << " SOct:" << anIm.ScaleInOct() ;

       tBase aVMax;
       ELISE_COPY(aTIm.all_pts(),aTIm.in(0),VMax(aVMax));
       ELISE_COPY(aWTesD.all_pts(), aTIm.in(0) * (255.0/aVMax) ,aWTesD.ogray());
       aWTesD.clik_in();

       std::cout << "\n";

       if ((aK>=1) && (aK<int(aVIm.size()-2)))
       {
       }
  }

}

void TestDigeoExt()
{
    std::string aName = "/home/marc/TMP/Delphes/12-Tetes-Inc-4341-6/IMG_0057.CR2";
    cParamAppliDigeo aParam;
    aParam.mSauvPyram = true;

    cAppliDigeo * anAD = DigeoCPP(aName,aParam);
    cImDigeo &  anImD = anAD->SingleImage(); // Ici on ne mape qu'une seule image à la fois


    std::cout << "Nb Box to do " << anAD->NbInterv() << "\n";
    for (int aKBox = 0 ; aKBox<anAD->NbInterv() ; aKBox++)
    {
        anAD->LoadOneInterv(aKBox);  // Calcul et memorise la pyramide gaussienne
        
        if (aKBox==0)
        {
            const std::vector<cOctaveDigeo *> & aVOct = anImD.Octaves();
            std::cout <<  "= Nombre Octaves " << aVOct.size() << "\n";
            for (int aKo=0 ; aKo<int(aVOct.size()) ; aKo++)
            {
                 cOctaveDigeo & anOct = *(aVOct[aKo]);
                 const std::vector<cImInMem *> & aVIms = anOct.VIms();
                 std::cout << " *Oct=" << aKo << " Dz=" << anOct.Niv()  << " NbIm " << aVIms.size();

                  cTplOctDig<U_INT2> * aUI2_Oct = anOct.U_Int2_This();  // entre aUI2_OctaUI2_Oct et  aR4_Oct
                  cTplOctDig<REAL4> * aR4_Oct = anOct.REAL4_This();     // un et un seul doit etre != 0

                 if (aUI2_Oct !=0) std::cout << " U_INT2 ";
                 if (aR4_Oct !=0) std::cout << " REAL4 ";
                 
                 std::cout << "\n";
                  
                 if (aUI2_Oct !=0) 
                    Show_Octave<U_INT2,INT>(aUI2_Oct);
                 if (aR4_Oct !=0) 
                    Show_Octave<REAL4,REAL8>(aR4_Oct);
            }
        }
        std::cout << "Done " << aKBox << " on " << anAD->NbInterv() << "\n";
    }
}





/*
        bool     mExigeCodeCompile;
        int      mNivFloatIm;        // Ne depend pas de la resolution
*/


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
