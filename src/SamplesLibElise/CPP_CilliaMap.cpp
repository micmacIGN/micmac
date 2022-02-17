/*Header-MicMac-eLise-25/09/2007
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

int CilliaMap_main(int argc, char ** argv)

{


        std::string aFile1;
        std::string aFile2;
        std::string aFileOut;


        ElInitArgMain
        (
            argc ,argv ,
            LArgMain() <<EAMC(aFile1, "aFile1" )
                       <<EAMC(aFile2, "aFile2")
                       <<EAMC(aFileOut, "fichier de sorti"),

            LArgMain()
        );
        
          cXml_Map2D  aMaps1,aMaps2,aMaps;
          aMaps1=  StdGetFromSI(aFile1,Xml_Map2D);
          aMaps2= StdGetFromSI(aFile2,Xml_Map2D);
          std::list< cXml_Map2DElem > aMapEL1;
          std::list< cXml_Map2DElem >::iterator itME1;
          std::list< cXml_Map2DElem > aMapEL2;
          std::list< cXml_Map2DElem >::iterator itME2;

          aMapEL1= aMaps1.Maps();

          itME1 = aMapEL1.begin();

          double  aSc1=0.0,aTrx1=0.0,aTry1=0.0; 
          double aSc2=0.0,aTrx2=0.0,aTry2=0.0;          

          for( ; itME1 != aMapEL1.end(); itME1++)

               {

                cTplValGesInit< cXml_Homot > aHomTmp1 = itME1->Homot();

                cXml_Homot  aHom1 = aHomTmp1.Val();


               aSc1= aHom1.Scale();
               aTrx1= aHom1.Tr().x;
               aTry1= aHom1.Tr().y;
   
              }

      aMapEL2= aMaps2.Maps();

     itME2 = aMapEL2.begin();
     for( ; itME2 != aMapEL2.end(); itME2++)
        {

          cTplValGesInit< cXml_Homot > aHomTmp2 = itME2->Homot();
          cXml_Homot  aHom2 = aHomTmp2.Val();

           aSc2= aHom2.Scale();
           aTrx2= aHom2.Tr().x;
           aTry2= aHom2.Tr().y;

        }
   
     cXml_Map2D aMap;
     cXml_Map2DElem aMapEL;
     std::list<cXml_Map2DElem> aLMEL;
     cXml_Homot Hom;         

     std::cout << Hom.Scale() << "\n";

     double aSc3= aSc1 * aSc2;
     Hom.Scale()=aSc3;    
      
     Pt2dr aTrn3;
    
    double  x = aTrx1 + aTrx2;
    double  y = aTry1 + aTry2;
        Hom.Tr() = Pt2dr(x,y);    
        aMapEL.Homot() = Hom;

       aLMEL.push_back(aMapEL);           
       aMap.Maps() = aLMEL;
   
       MakeFileXML(aMap, aFileOut);

return(1);
}

int SimilComp_main(int argc, char ** argv)

{
        std::string aFile1;
        std::string aFile2;
        std::string aFileOut;
        ElInitArgMain
        (
            argc ,argv ,
            LArgMain() <<EAMC(aFile1, "aFile1" )
                       <<EAMC(aFile2, "aFile2")
                       <<EAMC(aFileOut, "fichier de sorti"),

            LArgMain()
        );

          cXml_Map2D  aMaps1,aMaps2,aMaps;
          aMaps1=  StdGetFromSI(aFile1,Xml_Map2D);
          aMaps2= StdGetFromSI(aFile2,Xml_Map2D);
          std::list< cXml_Map2DElem > aMapEL1;
          std::list< cXml_Map2DElem >::iterator itME1;
          std::list< cXml_Map2DElem > aMapEL2;
          std::list< cXml_Map2DElem >::iterator itME2;

          aMapEL1= aMaps1.Maps();

          itME1 = aMapEL1.begin();

          double  aScx1=0.0; double aScy1=0.0 ; double aTrx1= 0.0 ,aTry1=0.0;
          double aScx2=0.0,aScy2=0.0,aTrx2=0.0,aTry2=0.0;

        for( ; itME1 != aMapEL1.end(); itME1++)

             {

          cTplValGesInit< cSimilitudePlane > aSimTmp1 = itME1->Sim();

          cSimilitudePlane  aSim1 = aSimTmp1.Val();

          aScx1= aSim1.Scale().x;
          aScy1= aSim1.Scale().y;
          aTrx1= aSim1.Trans().x;
          aTry1= aSim1.Trans().y;
       

            }

            aMapEL2= aMaps2.Maps();

     itME2 = aMapEL2.begin();
     for( ; itME2 != aMapEL2.end(); itME2++)
        {

          cTplValGesInit< cSimilitudePlane > aSimTmp2 = itME2->Sim();
          cSimilitudePlane  aSim2 = aSimTmp2.Val();
 
          aScx2= aSim2.Scale().x;
          aScy2= aSim2.Scale().y;
          aTrx2= aSim2.Trans().x;
          aTry2= aSim2.Trans().y;

         }
      cXml_Map2D aMap;
      cXml_Map2DElem aMapEL;
      std::list<cXml_Map2DElem> aLMEL;
      cSimilitudePlane Sim;

 

      double aScx3= aScx1 * aScx2;
      double aScy3=  aScy1 * aScy2;
     
      Sim.Scale().x=aScx3;
      Sim.Scale().y=aScy3;
      Pt2dr aTrn3;

      double  x = aTrx1 + aTrx2;
      double  y = aTry1 + aTry2;
      
       Sim.Trans() = Pt2dr(x,y);
       aMapEL.Sim() = Sim;

       aLMEL.push_back(aMapEL);
       aMap.Maps() = aLMEL;



       MakeFileXML(aMap, aFileOut);

return(1);
}
int AffineComp_main(int argc, char ** argv)

{
        std::string aFile1;
        std::string aFile2;
        std::string aFileOut;
        ElInitArgMain
        (
            argc ,argv ,
            LArgMain() <<EAMC(aFile1, "aFile1" )
                       <<EAMC(aFile2, "aFile2")
                       <<EAMC(aFileOut, "fichier de sorti"),

            LArgMain()
        );

         cXml_Map2D  aMaps1,aMaps2,aMaps;
         aMaps1=  StdGetFromSI(aFile1,Xml_Map2D);
         aMaps2= StdGetFromSI(aFile2,Xml_Map2D);
         std::list< cXml_Map2DElem > aMapEL1;
         std::list< cXml_Map2DElem >::iterator itME1;
         std::list< cXml_Map2DElem > aMapEL2;
         std::list< cXml_Map2DElem >::iterator itME2;

         aMapEL1= aMaps1.Maps();

         itME1 = aMapEL1.begin();

         double  aScx1=0.0;
         double aScy1=0.0; double aTrx1=0.0; double aTry1=0.0; double aRotx1=0.0; double  aRoty1=0.0;
         double aScx2=0.0; double aScy2=0.0; double aTrx2=0.0; double aTry2= 0.0; double  aRotx2=0.0 ; double aRoty2=0.0;

      for( ; itME1 != aMapEL1.end(); itME1++)

             {
 
              cTplValGesInit< cAffinitePlane  > aAffTmp1 = itME1->Aff();

              cAffinitePlane   aAff1 = aAffTmp1.Val();

             aTrx1= aAff1.I00().x;
             aTry1= aAff1.I00().y;
             aScx1= aAff1.V10().x;
             aScy1= aAff1.V10().y;
             aRotx1=aAff1.V01().x;
             aRoty1=aAff1.V01().y ; 
             }

     aMapEL2= aMaps2.Maps();

     itME2 = aMapEL2.begin();
     for( ; itME2 != aMapEL2.end(); itME2++)
        {

          cTplValGesInit< cAffinitePlane > aAffTmp2 = itME2->Aff();
          cAffinitePlane  aAff2 = aAffTmp2.Val();

         aTrx2= aAff2.I00().x;
         aTry2= aAff2.I00().y;
         aScx2= aAff2.V10().x;
         aScy2= aAff2.V10().y;
         aRotx2=aAff2.V01().x;
         aRoty2=aAff2.V01().y;

        }
  
      cXml_Map2D aMap;
      cXml_Map2DElem aMapEL;
      std::list<cXml_Map2DElem> aLMEL;
      cAffinitePlane Aff;
      double  aScx3= aScx1 * aScx2;
      double aScy3=  aScy1 * aScy2;
     
     Aff.V10()= Pt2dr(aScx3, aScy3);
     
     Pt2dr aTrn3;

     double  x = aTrx1 + aTrx2;
     double  y = aTry1 + aTry2;
            Aff.I00() = Pt2dr(x,y);

     double Rx = aRotx1 * aRotx2;
     double  Ry = aRoty1 * aRoty2;
        Aff.V01() = Pt2dr(Rx,Ry);
        aMapEL.Aff() = Aff;

        aLMEL.push_back(aMapEL);
        aMap.Maps() = aLMEL;



       MakeFileXML(aMap, aFileOut);

return(1);
}
