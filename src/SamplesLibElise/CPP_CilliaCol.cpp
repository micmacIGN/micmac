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


int ImgCol_main(int argc, char ** argv)

{


        std::string aImge1;
        std::string aIm2;
        std::string aIm3;
        std::string aIm4;
        std::string aLigne, aColone;
        std::string aNameMap;
        std::string aNameOut;

 
        ElInitArgMain
        (
            argc ,argv ,
            LArgMain() <<EAMC(aImge1, "image 1" )                  
                           <<EAMC(aIm2, "image 2" )
                         <<EAMC(aIm3, "image 3" )
                         <<EAMC(aIm4, "image 4" )
                         <<EAMC(aLigne, "taille de l'image" )
                         <<EAMC(aColone, "taille de l'image" )
                       <<EAMC(aNameMap, "Map initiale")
                       <<EAMC(aNameOut, "image de sorti"),
                    
            LArgMain()
        );
         
   /****1er image***/


         std::string  aCom= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aImge1 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + aNameMap +BLANK
                       + aNameOut ;
         system_call(aCom.c_str());
      
      
      int transformation;
      std::cout <<" enter le genre de la transformation 1 pour Homotheté 2 pour Similitude et 3 Affinité" <<"\n";
      std::cin >> transformation;

      std::string XmlOutFile,XmlOutFile1,XmlOutFile2,aXml, aXml1;
      std::string aCom1, aCom2 , aCom3 , aCom4, aCom5, aCom6, aComMap, aComMap1;





     switch (transformation){

     case 1 : 
   
   

/** Calcuel de transformation et insertion de la 2 em image**/
          XmlOutFile="Homot.xml" ;
          aCom1= MMBinFile("mm3d CalcMapAnalytik ")
                       + aImge1 +BLANK
                       + aIm2 +BLANK
                       +  "Homot "+BLANK
                       + XmlOutFile
                             ;
         system_call(aCom1.c_str());
                      
         aCom2= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aIm2 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + XmlOutFile +BLANK
                       + aNameOut ;
         system_call(aCom2.c_str());


/** Calcuel de transformation et insertion de la 3 em image**/


         XmlOutFile1="Homot1.xml" ;
         aCom3= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm2 +BLANK
                       + aIm3 +BLANK
                       +  "Homot "+BLANK
                       + XmlOutFile1
                             ;
         system_call(aCom3.c_str());
        
   
    
         aComMap= MMBinFile("mm3d Testlib TestCilliaMap ")
                         + XmlOutFile + BLANK
                         + XmlOutFile1 + BLANK
                         + "Homo31.xml";

         system_call(aComMap.c_str());
      

  
         aXml="Homo31.xml";
         aCom4= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aIm3 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + aXml +BLANK
                       + aNameOut ;
         system_call(aCom4.c_str());
         
    
/***Calcule de transformation et insertion de la 4 eme image**/

         XmlOutFile2="Homot3.xml" ;
         aCom5= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm3 +BLANK
                       + aIm4 +BLANK
                       +  "Homot "+BLANK
                       + XmlOutFile2
                             ;
         
         system_call(aCom5.c_str());

         aComMap1= MMBinFile("mm3d Testlib TestCilliaMap ")
                  + XmlOutFile2 + BLANK
                  + aXml + BLANK
                  + "Homo41.xml";

         system_call(aComMap1.c_str());


         aXml1="Homo41.xml";
         aCom6 = MMBinFile("mm3d Testlib TestCilliaAss ")
                        + aIm4 +BLANK
                        + aLigne +BLANK
                        + aColone   +BLANK
             	        + aXml1+BLANK
         	        + aNameOut ;
         system_call(aCom6.c_str());

break;
 

case 2 : 
         XmlOutFile="Simil.xml" ;
         aCom1= MMBinFile("mm3d CalcMapAnalytik ")
                       + aImge1 +BLANK
                       + aIm2 +BLANK
                       +  "Simil "+BLANK
                       + XmlOutFile
                             ;
         
         system_call(aCom1.c_str());

         aCom2= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aIm2 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + XmlOutFile +BLANK
                       + aNameOut ;
         system_call(aCom2.c_str());


/** Calcuel de transformation et insertion de la 3 em image**/


         XmlOutFile1="Simil1.xml" ;
         aCom3= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm2 +BLANK
                       + aIm3 +BLANK
                       +  "Simil "+BLANK
                       + XmlOutFile1
                             ;
          
         system_call(aCom3.c_str());

       
         aComMap= MMBinFile("mm3d Testlib SimilComp ")
                          + XmlOutFile + BLANK
                          + XmlOutFile1 + BLANK
                          + "Simil31.xml";

         system_call(aComMap.c_str());


         aXml="Simil31.xml";
         aCom4 = MMBinFile("mm3d Testlib TestCilliaAss ")
                        + aIm3 +BLANK
                        + aLigne +BLANK
                        + aColone   +BLANK
                        + aXml +BLANK
                        + aNameOut ;
         system_call(aCom4.c_str());



/***Calcule de transformation et insertion de la 4 eme image**/

         XmlOutFile2="Simil3.xml" ;
         aCom5= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm3 +BLANK
                       + aIm4 +BLANK
                       +  "Simil "+BLANK
                       + XmlOutFile2
                             ;
         std::cout<< "la commande " << aCom6 <<"\n";

         system_call(aCom5.c_str());
          
         
         aComMap1= MMBinFile("mm3d Testlib SimilComp ")
                          + XmlOutFile2 + BLANK
                          + aXml + BLANK
                          + "Simil41.xml";

         system_call(aComMap1.c_str());

         aXml1="Simil41.xml";
         
         aCom6= MMBinFile("mm3d Testlib TestCilliaAss ")
                         + aIm4 +BLANK
                         + aLigne +BLANK
                         + aColone   +BLANK
                         + aXml1+BLANK
                         + aNameOut ;
         system_call(aCom6.c_str());
break;

case 3: 
         
         XmlOutFile="Affine.xml" ;
         aCom1= MMBinFile("mm3d CalcMapAnalytik ")
                       + aImge1 +BLANK
                       + aIm2 +BLANK
                       +  "Affine "+BLANK
                       + XmlOutFile
                             ;
         system_call(aCom1.c_str());

         aCom2= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aIm2 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + XmlOutFile +BLANK
                       + aNameOut ;
         system_call(aCom2.c_str());


/** Calcuel de transformation et insertion de la 3 em image**/


         XmlOutFile1="Affine1.xml" ;
         aCom3= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm2 +BLANK
                       + aIm3 +BLANK
                       +  "Affine "+BLANK
                       + XmlOutFile1
                             ;
         system_call(aCom3.c_str());

           // aXml="Homo31.xml";  
         aComMap= MMBinFile("mm3d Testlib AffineComp ")
                           + XmlOutFile + BLANK
                           + XmlOutFile1 + BLANK
                           + "Affine31.xml";

         system_call(aComMap.c_str());


        
         aXml="Affine31.xml";
         aCom4= MMBinFile("mm3d Testlib TestCilliaAss ")
                       + aIm3 +BLANK
                       + aLigne +BLANK
                       + aColone   +BLANK
                       + aXml +BLANK
                       + aNameOut ;
         system_call(aCom4.c_str());


/***Calcule de transformation et insertion de la 4 eme image**/

   
         XmlOutFile2="Affine3.xml" ;
         aCom5= MMBinFile("mm3d CalcMapAnalytik ")
                       + aIm3 +BLANK
                       + aIm4 +BLANK
                       +  "Affine"+BLANK
                       + XmlOutFile2
                             ;
        
         system_call(aCom5.c_str());
         aComMap1= MMBinFile("mm3d Testlib AffineComp ")
                             + XmlOutFile2 + BLANK
                             + aXml + BLANK
                             + "Affine41.xml";

         system_call(aComMap1.c_str());


         aXml1="Affine41.xml";
         aCom6= MMBinFile("mm3d Testlib TestCilliaAss ")
                        + aIm4 +BLANK
                        + aLigne +BLANK
                        + aColone   +BLANK
                        + aXml1+BLANK
                        + aNameOut ;
         system_call(aCom6.c_str());


         
}
return(1);

}
