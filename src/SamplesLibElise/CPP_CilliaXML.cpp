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
#if (ELISE_MacOs)
#define SSTR( x ) (std::ostringstream()<< std::dec << x).str()
#else
#define SSTR( x ) static_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()
#endif

int HomToXML_main( int argc , char ** argv)
{
      std::string aFile1;
      std::string aFileXml;
      std::string aEchelle;
      	
      ElInitArgMain
        (
          argc, argv,
          LArgMain()  << EAMC(aFile1,"fichier.txt")
                      << EAMC(aFileXml,"fichier xml"),
          LArgMain()  << EAM(aEchelle,"echelle",true,"conefficient du sous-echantionnage")
        );
        

       //Déclartion des différentes classes
       cSetOfMesureAppuisFlottants aSetMAF;
       cMesureAppuiFlottant1Im aMAF; 
       cMesureAppuiFlottant1Im aMAF1;
       cOneMesureAF1I  aOM;
       cOneMesureAF1I  aOM1;       
      
       //lecture du fichier.txt
       ELISE_fp aFIn(aFile1.c_str(),ELISE_fp::READ);
       char *  aLine;
       aLine = aFIn.std_fgets();

       char  nom1[50];
       char  nom2[50]; 

      
        sscanf(aLine,"%s %s", nom1, nom2);  
         
        std::cout << nom1 << "\n" ; 
     
        aMAF.NameIm()=nom1;
        aMAF1.NameIm()=nom2;

         
     	
        int Ccnt=0;   
   while ((aLine = aFIn.std_fgets()))
       {
          Ccnt ++;
             
          std::string Cnt = SSTR( Ccnt );
          std::cout << "" << Cnt << "\n";  

          std::vector<Pt4dr> aV4Ok;
          Pt4dr aP;  
           
          int aNb = sscanf(aLine,"%lf  %lf  %lf %lf ",&aP.x , &aP.y , &aP.z , &aP.t);  
          ELISE_ASSERT(aNb==4,"Could not read 4 double values");

          aV4Ok.push_back(aP); 	
 
          Pt2dr aPP,aPP1;

         if (EAMIsInit(&aEchelle))
             {
            	int x = atoi( aEchelle.c_str()  ); 
       
            	Pt2dr  aPP(aP.x*x,aP.y*x );
            	Pt2dr  aPP1(aP.z*x,aP.t*x ); 
        
       	        std::string str("point");      
        
            	aOM.NamePt()=str+Cnt;
            	aOM1.NamePt()=str+Cnt;
            	aOM.PtIm()=aPP;
                aOM1.PtIm()=aPP1;	
             }
         else		
             {
               Pt2dr aPP(aP.x,aP.y);
               Pt2dr aPP1(aP.z,aP.t);       
        
               std::string str("point");      
        
               aOM.NamePt()=str+Cnt;
               aOM1.NamePt()=str+Cnt;
               aOM.PtIm()=aPP;
               aOM1.PtIm()=aPP1;
             }    
              aMAF.OneMesureAF1I().push_back(aOM);
              aMAF1.OneMesureAF1I().push_back(aOM1);
        }

   
    aSetMAF.MesureAppuiFlottant1Im().push_back(aMAF);
    aSetMAF.MesureAppuiFlottant1Im().push_back(aMAF1);
    MakeFileXML(aSetMAF,aFileXml);

return(1) ;

}
