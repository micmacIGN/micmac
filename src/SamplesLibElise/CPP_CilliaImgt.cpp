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
 
int CilliaImgt_main(int argc, char ** argv)
{
 
        std::string aImg1;
        std::string aImg2;
        std::string aXmlFile;
        std::string aLigne;
        std::string aColone;
         
          
        ElInitArgMain
        (
            argc ,argv ,
            LArgMain() <<EAMC(aImg1, "image 1" )
                       <<EAMC(aImg2, "image 2" )
                       <<EAMC(aXmlFile, "homorg.xml" )
                       <<EAMC(aLigne, "homorg.xml" )
                       <<EAMC(aColone, "homorg.xml" ),
            LArgMain()
        );
        
/*** Récuperer le nobre de processeur **/
      int aNb= NbProcSys();
     
    std::string mDir, mPat; 
  // cInterfChantierNameManipulateur * mICNM;
  
  

    Tiff_Im  ImgTiff= Tiff_Im::UnivConvStd(aImg1.c_str());

    Im2D<INT4,INT> I(ImgTiff.sz().x, ImgTiff.sz().y);
    ELISE_COPY
    (
        I.all_pts(),
        ImgTiff.in(),
        I.out()
    );

   
  //std::cout << ImgTiff.sz() <<"\n";

    std::string aNom,aNom1;

    int x1= atoi( aLigne.c_str()  );
    int y1=  atoi( aColone.c_str()  );
    Pt2di aSz(x1 ,y1);


    cElMap2D * aMaps ;
    aMaps = cElMap2D::FromFile(aXmlFile);


    Tiff_Im  ImgTiff2= Tiff_Im::UnivConvStd(aImg2.c_str());
    Im2D<INT4,INT> I2(ImgTiff2.sz().x, ImgTiff2.sz().y);
 

    ELISE_COPY
       (
        I2.all_pts(),
        ImgTiff2.in(),
        I2.out()
       );
    
     std::pair<std::string, std::string > tPair;  
     std::vector<std::pair<std::string,std::string> > Vec,Vec1, Vec2;
   
    std::string  aCom;
    std::list<std::string> aLCom;
 
    int aNbc;//nombre de couple d'image à appliquer la commande Tapioca en parallel ;
    int i;
    i=1;
    aNbc=aNb/2;

 
  for (int j=1 ; j<=8 ;j++) // pour parcourir toute l'image
     {
    
        for (i=i; i<=aNbc ; i++ )

          {
           std::string Cnt = SSTR( i ); 
           aNom="im"+Cnt+".""tif";
 
           Tiff_Im   aTOut
                 (
                aNom.c_str(),
                aSz,
                ImgTiff.type_el(),
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
                 );

           ELISE_COPY
               (
             rectangle(Pt2di(0,0),Pt2di(x1,y1)),
             trans( I.in(),Pt2di((i-1)*x1,0)),
             aTOut.out()
              );

           Pt2di aPaa((i-1)*x1 ,0);
           Pt2di aaPa1(i*x1,y1);
           Pt2dr aQaa = (*aMaps)(Pt2dr(aPaa));
           Pt2dr aQ1aa =  (*aMaps)(Pt2dr(aaPa1));
   
           int aQaaxi;
           if(aQaa.x >0)// pour verifier si la coordonées x  est bien dasn l'image 
              {
               aQaaxi=floor(aQaa.x);
              }
           else
            {
               aQaaxi=0 ;
            } 
           int aQaayi;
           if(aQaa.y >0)// pour vérifier si la coordonnée y est bien dans l'image
               { 
                 aQaayi=floor(aQaa.y);
               }   
           else   
              {
            aQaayi=0 ;
              }
           int aQ1aaxi=floor(aQ1aa.x);
           int aQ1aayi=floor(aQ1aa.y);
           Pt2di aPPaa(aQaaxi,aQaayi);
           Pt2di aPP1aa(aQ1aaxi,aQ1aayi);

           aNom1="imm"+Cnt+".""tif";
           Tiff_Im  aTOut1
                 (
                  aNom1.c_str(),
                  aSz,
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  Tiff_Im::BlackIsZero
                );

           ELISE_COPY
               (
                rectangle(Pt2di(0,0) ,Pt2di(aSz.x,aSz.y)),
                trans(I2.in(),Pt2di(aPPaa)),
                aTOut1.out()
               );

           aCom= MMBinFile("mm3d Tapioca All ")
                       + "\"("+ aNom +"|" + aNom1 +")\"" +   BLANK
                       + "-1 " +BLANK + "ByP=1"
                       ;

      
           aLCom.push_back(aCom);
  
         
           tPair.first=aNom;
           tPair.second=aNom1;
    
           Vec.push_back(tPair);
             
         }
   cEl_GPAO::DoComInParal(aLCom);
   aNbc=i+3; 
       
}
   

/*****
   Deuxième rangé        **/

   int n,m;
   n=1;
   m=aNb/2;
       
  for (int j=1 ; j<=8 ;j++)
     {
     for (n=n; n<=m ; n++ )
          {
             std::string Cnt = SSTR( n );
             aNom="img"+Cnt+".""tif";

             Tiff_Im  aTOut
                      (
                     aNom.c_str(),
                     aSz,
                     ImgTiff.type_el(),
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero
                      );

             ELISE_COPY
                     (
                      rectangle(Pt2di(0,0),Pt2di(x1,2*y1)),
                      trans( I.in(),Pt2di((n-1)*x1,y1)),
                      aTOut.out()
                    );

            Pt2di aPaa((n-1)*x1 ,y1);
            Pt2di aaPa1(n*x1,2*y1);
            Pt2dr aQaa = (*aMaps)(Pt2dr(aPaa));
            Pt2dr aQ1aa =  (*aMaps)(Pt2dr(aaPa1));
  

            int aQaaxi;
            if(aQaa.x >0)//pour vérifier si la coordonnée x est bien à l'interieure de l'image
                    {
                      aQaaxi=floor(aQaa.x);
                    }
            else
               {
                     aQaaxi=0 ;
               }

           int aQaayi;
           if(aQaa.y >0)//pour vérifier que la coordonnées y est bien à l'interrieure de l'image
                   {
                   aQaayi=floor(aQaa.y);
                   }
           else
              {
            aQaayi=0 ;
              }


           int aQ1aaxi=floor(aQ1aa.x);
           int aQ1aayi=floor(aQ1aa.y);

           Pt2di aPPaa(aQaaxi,aQaayi);
           Pt2di aPP1aa(aQ1aaxi,aQ1aayi);

           aNom1="imgg"+Cnt+".""tif";


           Tiff_Im  aTOut1
                (
                     aNom1.c_str(),
                     aSz,
                     GenIm::u_int1,
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero
                );

           ELISE_COPY
                 (
                 rectangle(Pt2di(0,0) ,Pt2di(aSz.x,aSz.y)),
                 trans(I2.in(),Pt2di(aPPaa)),
                 aTOut1.out()
                 );



           aCom= MMBinFile("mm3d Tapioca All ")
                       + "\"("+ aNom +"|" + aNom1 +")\"" +   BLANK
                       + "-1 " +BLANK + "ByP=1"
                       ;

           aLCom.push_back(aCom);

         tPair.first=aNom;
         tPair.second=aNom1;

         Vec1.push_back(tPair);
             
         }
     cEl_GPAO::DoComInParal(aLCom);
     m=n+3;

    }
      
/******
       Troisieme range ***////
  //  int y=1075;     
       int c,l;
       c=1;
       l=aNb/2;

  for (int j=1 ; j<=8 ;j++)
     {
        for (c=c; c<=l ; c++ )
          {
         std::string Cnt = SSTR( c );
         aNom="imge"+Cnt+".""tif";

         Tiff_Im aTOut
               (
                   aNom.c_str(),
                   aSz,
                   ImgTiff.type_el(),
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero
                );

         ELISE_COPY
              (
               rectangle(Pt2di(0,0),Pt2di(x1,y1)),
               trans( I.in(),Pt2di((c-1)*x1,2*y1)),
               aTOut.out()
               );

         Pt2di aPaa((c-1)*x1 ,2*y1);

  
         Pt2di aaPa1(c*x1,3*y1);
         Pt2dr aQaa = (*aMaps)(Pt2dr(aPaa));
         Pt2dr aQ1aa =  (*aMaps)(Pt2dr(aaPa1));
   
 
         int aQaaxi;
         if(aQaa.x >0)
            { 
               aQaaxi=floor(aQaa.x);
         
            }  
         else
            {
            aQaaxi=0 ;
            }
   

         int aQaayi;
           if(aQaa.y >0)
               {
                  aQaayi=floor(aQaa.y);

               }
         else
             {
              aQaayi=0 ;
             }

         int aQ1aaxi=floor(aQ1aa.x);
         int aQ1aayi=floor(aQ1aa.y);


         Pt2di aPPaa(aQaaxi,aQaayi);
         Pt2di aPP1aa(aQ1aaxi,aQ1aayi);

         aNom1="imgge"+Cnt+".""tif";


         Tiff_Im     aTOut1
                 (
                   aNom1.c_str(),
                   aSz,
                   GenIm::u_int1,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero
                  );

         ELISE_COPY
               (
                 rectangle(Pt2di(0,0) ,Pt2di(x1,y1)),
                 trans(I2.in(),Pt2di(aPPaa)),
                 aTOut1.out()
               );



         aCom= MMBinFile("mm3d Tapioca All ")
                       + "\"("+ aNom +"|" + aNom1 +")\"" +   BLANK
                       + "-1 " +BLANK + "ByP=1"
                       ;

     
         aLCom.push_back(aCom);
         tPair.first=aNom;
         tPair.second=aNom1;

         Vec2.push_back(tPair);
           }


    cEl_GPAO::DoComInParal(aLCom);
    l=c+3;

}




          std::string mFullName= "/home/cillia/Bureau/a-transferer/Data/Dimage/DADF/.*tif";   
          SplitDirAndFile(mDir,mPat,mFullName);
          std::cout << mPat <<"\n";
   //cInterfChantierNameManipulateur *mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
          std::cout << "directory" << mDir << "\n";

          ElPackHomologue aPckOut , aPckOut1;
          std::string  aHmOut =mDir + "Homol1" ;
          ELISE_fp::MkDir( aHmOut );


          std::string mPas= aHmOut +"/Pastis" + aImg1;
          ELISE_fp::MkDir( mPas );
          std::string mPas1= aHmOut +"/Pastis" + aImg2;
          ELISE_fp::MkDir( mPas1 );
 
          std::string aHomOut= mPas + "/" + aImg2 + ".txt";
          std::string aHomOut1= mPas1 + "/" + aImg1 + ".txt";

            
       for(
	          int aS1=0;
                  aS1<int(Vec.size());
                  aS1++
           )
                 
                  {    
                  
                  
                   std::string mPatHom = mDir + "Homol/Pastis" + Vec.at(aS1).first + "/" + Vec.at(aS1).second + ".dat";
    
                   std::string mPatHom2 = mDir + "Homol/Pastis" + Vec.at(aS1).second + "/" + Vec.at(aS1).first + ".dat"  ;                                              

                   if( ELISE_fp::exist_file(mPatHom))
                             {
                                 
                             ElPackHomologue aPack = ElPackHomologue::FromFile(mPatHom);
                             for (
                                           ElPackHomologue::iterator iTH=aPack.begin();
                                           iTH!=aPack.end();
                                           iTH++
                                  )
                              { 
                                          Pt2dr aP1 = iTH->P1();
                                          Pt2dr aP2 = iTH->P2();                       
                                        double aP= aP1.x + aS1 * x1;
                                        int aIn=floor(aP);
                                        int aIn1=floor(aP1.y);                                     
                                        Pt2dr aPP1(aIn,aIn1);
                                        
                                        double aPP= aP2.x + aS1*x1;
                                        int aIn2=floor(aPP);
                                        int aIn22=floor(aP2.y);
                                        Pt2dr aPP2(aIn2, aIn22);
                                             
                                  aPckOut.Cple_Add(ElCplePtsHomologues (aPP1,aPP2));
                                }
                                
                                     aPckOut.StdPutInFile(aHomOut);                                                                                                     
                            }          
                                                                    
                              else {
                                           std::cout  << "le fichier n'existe pas " <<"\n";
                            
                                    }

                                   
                      }


            for(
                          int aS1=0;
                          aS1<int(Vec1.size());
                          aS1++
           )
                  {

                   std::cout << "first" << Vec1.at(aS1).first <<"\n";

                  std::string mPatHom1 = mDir + "Homol/Pastis" + Vec1.at(aS1).first + "/" + Vec1.at(aS1).second + ".dat";
                  std::string mPatHom2 = mDir + "Homol/Pastis" + Vec1.at(aS1).second + "/" + Vec1.at(aS1).first + ".dat" ;
                   
                   std::cout<<" mPatHom" << mPatHom1 <<"\n";


            if( ELISE_fp::exist_file(mPatHom1))
                             {


                             ElPackHomologue aPack1 = ElPackHomologue::FromFile(mPatHom1);
                            // std::cout << " pppp"<< "\n";  
                             for (
                                           ElPackHomologue::iterator iTH=aPack1.begin();
                                           iTH!=aPack1.end();
                                           iTH++
                                        )
                              {
                                          Pt2dr aP1 = iTH->P1();
                                          Pt2dr aP2 = iTH->P2();
                                        //  std::cout <<" appp" << aP1.x << "\n";


                                        double aP= aP1.x + aS1 * x1;
                                        double aPP=aP1.y + y1;
                                         
                                        int aIn=floor(aP);
                                        int aIn1=floor(aPP);
                                        Pt2dr aPP1(aIn,aIn1);

                                        double aPPX= aP2.x + aS1*x1;
                                        double aPPY= aP2.y + y1;
                                        int aIn2=floor(aPPX);
                                        int aIn22=floor(aPPY);
                                        Pt2dr aPP2(aIn2, aIn22);

                                  aPckOut.Cple_Add(ElCplePtsHomologues (aPP1,aPP2));
                                }

                                     aPckOut.StdPutInFile(aHomOut);
                            }

                              else {
                                           std::cout  << "le fichier n'existe pas " <<"\n";

                                    }


                      }




        for(
                          int aS1=0;
                          aS1<int(Vec2.size());
                          aS1++
           )
                
                   {                
                    std::string mPatHom1 = mDir + "Homol/Pastis" + Vec2.at(aS1).first + "/" + Vec2.at(aS1).second + ".dat";
                    std::string mPatHom2 = mDir + "Homol/Pastis" + Vec2.at(aS1).second + "/" + Vec2.at(aS1).first + ".dat" ;
          
                   

                if( ELISE_fp::exist_file(mPatHom1) && ELISE_fp::exist_file(mPatHom2) )
                             {


                             ElPackHomologue aPack1 = ElPackHomologue::FromFile(mPatHom1);
                             ElPackHomologue aPack2 = ElPackHomologue::FromFile(mPatHom2);
                            // std::cout << " pppp"<< "\n";  
                             for (
                                           ElPackHomologue::iterator iTH=aPack1.begin();
                                           iTH!=aPack1.end();
                                           iTH++
                                        )
                              {
                                          Pt2dr aP1 = iTH->P1();
                                          Pt2dr aP2 = iTH->P2();
                                         // std::cout <<" appp" << aP1.x << "\n";


                                        double aP= aP1.x + aS1 * x1;
                                        double aPP=aP1.y + 2*y1;

                                        int aIn=floor(aP);
                                        int aIn1=floor(aPP);
                                        Pt2dr aPP1(aIn,aIn1);

                                        double aPPX= aP2.x + aS1*x1;
                                        double aPPY= aP2.y + y1;
                                        int aIn2=floor(aPPX);
                                        int aIn22=floor(aPPY);
                                        Pt2dr aPP2(aIn2, aIn22);
                                         aPckOut.Cple_Add(ElCplePtsHomologues (aPP1,aPP2));
                                }

                                     aPckOut.StdPutInFile(aHomOut);
                             


                              for (
                                           ElPackHomologue::iterator iTH1=aPack2.begin();
                                           iTH1!=aPack2.end();
                                           iTH1++ 
                                        ) 
                              {
                                          Pt2dr aP1 = iTH1->P1();
                                          Pt2dr aP2 = iTH1->P2();
                                         // std::cout <<" appp" << aP1.x << "\n";


                                        double aP= aP1.x + aS1 * x1;
                                        double aPP=aP1.y + 2*y1;

                                        int aIn=floor(aP);
                                        int aIn1=floor(aPP);
                                        Pt2dr aPP1(aIn,aIn1);

                                        double aPPX= aP2.x + aS1*x1;
                                        double aPPY= aP2.y + y1;
                                        int aIn2=floor(aPPX);
                                        int aIn22=floor(aPPY);
                                        Pt2dr aPP2(aIn2, aIn22);
                                         aPckOut.Cple_Add(ElCplePtsHomologues (aPP1,aPP2));
                                }

                                     aPckOut.StdPutInFile(aHomOut1);
                            }

                              else {
                                           std::cout  << "le fichier n'existe pas " <<"\n";

                                    }


                    }
return(1);
}
