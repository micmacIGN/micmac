/*Header-MicMac-eLiSe-25/06/2007

MiccMac : Multi Image Correspondances par Methodes Automatiques de Correlation
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
    eLiSCoordones_global_essai.txte image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.
Header-MicMac-eLiSe-25/06/2007*/

#include "StdAfx.h"

int  Homol2GCP_main(int argc,char ** argv)

{
          std::string aFile1;
          std::string aFile2;
          std::string aFileOut;

       ElInitArgMain
        (
            argc ,argv ,
                 LArgMain()
                       << EAMC(aFile1, "Sentinel Homol" )
                       << EAMC(aFile2, "tfw file")
                       << EAMC(aFileOut, "out= global coordinate file"),
                 LArgMain()
        );



        ELISE_fp aFIn(aFile1.c_str(),ELISE_fp::READ);
        ELISE_fp aFIn2(aFile2.c_str(),ELISE_fp::READ);
        FILE * aFOut = FopenNN(aFileOut.c_str(),"w","Cillia") ;
        char * aLine;
        std::vector<Pt2dr> aV2Ok;
     // std::vector<Pt3dr> aV3Ok;
        std::vector<double> tfw;



         while ((aLine = aFIn2.std_fgets()))
         {
               double  V ;
               int aNb=sscanf(aLine,"%lf ",&V );
               ELISE_ASSERT(aNb==1,"Could not read 2 double values");

               tfw.push_back(V);
         } 

        double tfw1 = tfw.at(0);
        double tfw4 = tfw.at(3);
	double tfw5 = tfw.at(4);
        double tfw6 = tfw.at(5);

        while ((aLine = aFIn.std_fgets()))
         {


            Pt2dr aP;
            int aNb = sscanf(aLine,"%lf %lf",&aP.x , &aP.y);
            ELISE_ASSERT(aNb==2,"Could not read 2 double values");

            aV2Ok.push_back(aP);

            fprintf(aFOut,"%lf %.14lf ", aP.x *tfw1 + tfw5 , aP.y *  tfw4 + tfw6 );
	}
 
return (1);

      
	delete aLine;
        delete aFOut;
         fclose(aFOut);


// return(1);
}
  int GlobToLocal_main(int argc,char ** argv)

   {
             
        std::string aFile1;
        std::string aFile2;
        std::string aFileOut;

       ElInitArgMain
        (
            argc ,argv ,
                LArgMain()
                     << EAMC(aFile1, "Global coordinate " )
                     << EAMC(aFile2, " tfw file")
                     << EAMC(aFileOut, "Out= Local coordinate file"),
               LArgMain()
        );



	

        ELISE_fp aFIn2(aFile2.c_str(),ELISE_fp::READ);



        char * aLine;
        std::vector<double> tfw;


         while ((aLine = aFIn2.std_fgets()))
         {         
             double  V ;
            int aNb=sscanf(aLine,"%lf ",&V );
            ELISE_ASSERT(aNb==1,"Could not read 2 double values")
                tfw.push_back(V);
         }
 	double tfw1 = tfw.at(0);
        double tfw4 = tfw.at(3);
        double tfw5 = tfw.at(4);
        double tfw6 = tfw.at(5);

        ELISE_fp aFIn(aFile1.c_str(),ELISE_fp::READ);
        FILE * aFOut = FopenNN(aFileOut.c_str(),"w","Cilliap") ;

        char * aLine2;
        std::vector<Pt2dr> aV2Ok;
	int aCnt=0;
        while ((aLine2 = aFIn.std_fgets()))
        {
      
	    aCnt++;
	
            Pt2dr aP;
           int aNb= sscanf(aLine2,"%lf  %lf",&aP.x , &aP.y);
            ELISE_ASSERT(aNb==2,"Could not read 2 double values");

            aV2Ok.push_back(aP);

           std::cout << aCnt << "  " << aP << "\n";      
            
	}
	

	for(int aK=0; aK<int(aV2Ok.size()-1); aK++)
	{
		fprintf(aFOut,"%.14lf %.14lf\n ", (aV2Ok.at(aK).x - tfw5)/ tfw1 , (aV2Ok.at(aK).y- tfw6) / tfw4 );
	}
	fprintf(aFOut,"%.14lf %.14lf ", (aV2Ok.at(aV2Ok.size()-1).x - tfw5)/ tfw1 , (aV2Ok.at(aV2Ok.size()-1).y- tfw6) / tfw4 );

	fclose(aFOut);	
        delete aLine;
        delete aLine2;

	return EXIT_SUCCESS;

}

int ExtractZ_main(int argc,char ** argv)


{
        std::string aFile1;
        std::string aImg;
        std::string aFileOut;

       ElInitArgMain
        (
            argc ,argv ,
                LArgMain()
                     << EAMC(aFile1, "Local coordinate " )
                     << EAMC(aImg, "SRTM image")
                     << EAMC(aFileOut, "Out= coordinate xyz file"),
               LArgMain()
        );

  // Read of an image
    Tiff_Im  ImgTiff = Tiff_Im::UnivConvStd(aImg.c_str());

    Im2D_REAL4 I(ImgTiff.sz().x, ImgTiff.sz().y);
    ELISE_COPY
    (
	I.all_pts(),
        ImgTiff.in(),
        I.out()  
    );

     //interpolation
     cInterpolBilineaire<REAL4> * bicu = new cInterpolBilineaire<REAL4>;
    //Lecture de point qui sont deja dans l'espace d'image SRTM"	 
     ELISE_fp aFIn(aFile1.c_str(),ELISE_fp::READ);
     std::vector<Pt2dr> aV2Ok;
     char * aLine;

     while ((aLine = aFIn.std_fgets()))
       {
            Pt2dr aP;
            int aNb=sscanf(aLine,"%lf  %lf",&aP.x , &aP.y);
            ELISE_ASSERT(aNb==2,"Could not read 2 double values");             

            aV2Ok.push_back(aP);

            //std::cout << "  " << aP << "\n";
       }

    // Pt2dr aP;

    // Recuperation de valeurs Z sur SRTM 
   
     FILE *aFOut = FopenNN(aFileOut.c_str(),"w","Cillia") ;
     double aZTmp=0;
     int  aCntOut=0;
   

     for(int aK=0; aK< int(aV2Ok.size()); aK++)
     {
	 if( (aV2Ok.at(aK).x >=0) && 
             (aV2Ok.at(aK).x <ImgTiff.sz().x) &&
	     (aV2Ok.at(aK).y >=0) &&
             (aV2Ok.at(aK).y <ImgTiff.sz().y) )
         {
             aZTmp = I.Get(aV2Ok.at(aK), *bicu , 0.5);

             fprintf(aFOut, "%lf %lf %lf \n", aV2Ok.at(aK).x, aV2Ok.at(aK).y, aZTmp);  
         }
	 else { aCntOut++; std::cout << "Point out of image=" << aV2Ok.at(aK) << " " << aCntOut << "\n"; }
     }

return EXIT_SUCCESS;

     fclose(aFOut);
}


int XYZ_Global_main(int argc,char ** argv)
{

        std::string aFile1;
        std::string aFile2;
        std::string aFileOut;

       ElInitArgMain
        (
            argc ,argv ,
                LArgMain()
                     << EAMC(aFile1, "Global coordinate " )
                     << EAMC(aFile2, " xyz file")
                     << EAMC(aFileOut, "Out= file XYZ global"),
               LArgMain()
        );
        ELISE_fp aFIn1(aFile1.c_str(),ELISE_fp::READ);
        ELISE_fp aFIn2(aFile2.c_str(),ELISE_fp::READ);
       // std::string aNameOut="XYZ_global_Full_final.txt";
       FILE * aFOut = FopenNN(aFileOut.c_str(),"w","Cillia") ;
       std::vector<Pt2dr> aV2Ok;
       char * aLine1;
       std::vector<Pt3dr> aV3Ok;
       char * aLine2;
       Pt3dr aP3d;
       Pt2dr aP2d;
       std::vector<Pt2dr>  aVXY;
       std::vector<double> aVZ;
        
      //read the first file
       while  ( (aLine1 = aFIn1.std_fgets())  )
          {

        
        int aNb1 = sscanf(aLine1,"%lf %lf", &aP2d.x , &aP2d.y);
        ELISE_ASSERT(aNb1==2,"Could not read 2 double values");

        aVXY.push_back(aP2d);

        std::cout << "AA=" << aP2d.x <<"\n" ;
           
        }

	//read the second file
	while ( (aLine2=aFIn2.std_fgets()) )
	{
	    int aNb2 = sscanf(aLine2,"%lf %lf %lf", &aP3d.x , &aP3d.y , &aP3d.z);
            ELISE_ASSERT(aNb2==3,"Could not read 3 double values");

	    aVZ.push_back(aP3d.z);
	}

	if( aVZ.size() == aVXY.size() ) 
		for( int aK=0; aK<int(aVZ.size()); aK++)
			fprintf(aFOut, "%lf %lf %lf\n",aVXY.at(aK).x  ,aVXY.at(aK).y , aVZ.at(aK) );
	else
	    ELISE_ASSERT(false,"The number of points in your files is not the same")

return(1);
}

