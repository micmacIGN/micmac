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
 
          
int CilliaAss_main(int argc, char ** argv)
{
        std::string aImge1;
        std::string aNameMap;
        std::string aLigne;
        std::string aColone;
        std::string aNameOut;
          
       
        ElInitArgMain
        ( 
            argc ,argv ,
            LArgMain() <<EAMC(aImge1, "image 1" )
                       <<EAMC(aLigne, "taille de l'image ")
                       <<EAMC(aColone, " taile de l'image")
                       <<EAMC(aNameMap, "Map initiale")
                       <<EAMC(aNameOut, "image de sorti"),
            LArgMain()
        );


 

      Tiff_Im  ImgTiff= Tiff_Im::UnivConvStd(aImge1.c_str());
      Tiff_Im  aTOut = Tiff_Im::StdConvGen(aNameOut,-1,true);
    

     std::list< cXml_Map2DElem > aMapEL;
     std::list< cXml_Map2DElem >::iterator itME;
     cXml_Map2D aMaps1; 
     cElMap2D * aMaps ;     
     std::vector<Im2DGen *> aVecImOut,aVecImIn;


    int x = atoi( aLigne.c_str()  );
    int y = atoi( aColone.c_str()  );

    Pt2di aSzOut(x,y);


/****appliquer la transformation ***/  
    aMaps = cElMap2D::FromFile(aNameMap);
    aVecImIn =  ImgTiff.ReadVecOfIm();
    int aNbC = aVecImIn.size();
   // Pt2di aSzIn = aVecImIn[0]->sz();
   
        if (!EAMIsInit(&aNameOut))
            aVecImOut =  ImgTiff.VecOfIm(aSzOut);
        else
            aVecImOut = aTOut.ReadVecOfIm();



    std::vector<cIm2DInter*> aVInter;
    for (int aK=0 ; aK<aNbC ; aK++)
    {
        aVInter.push_back(aVecImIn[aK]->SinusCard(3,3));
    }
   Pt2di aP;

    for (aP.x =0 ; aP.x<aSzOut.x ; aP.x++)
    {
        for (aP.y =0 ; aP.y<aSzOut.y ; aP.y++)
        {
            Pt2dr aQ = (*aMaps)(Pt2dr(aP));
            
            for (int aK=0 ; aK<aNbC ; aK++)
            {
                double aV = aVInter[aK]->GetDef(aQ,0);
                if( aV != 0 )
                                        aVecImOut[aK]->SetR(aP,aV);
                                else
                                {
                                        aVecImOut[aK]->SetR(aP,aVecImOut[aK]->GetR(aP,0));
                                }
            }
        }
    }



/***** inserer l'image ****/

    ELISE_COPY(aTOut.all_pts(),StdInPut(aVecImOut),aTOut.out());

return(1);
}   
