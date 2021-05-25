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


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/

int ANCIEN_TD_EXEMPLE_main(int argc,char ** argv)
{
    std::string aNameCam,aNameAppuis;
    std::string toto;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameCam,"Name of camera")
                    << EAMC(aNameAppuis,"Name of GCP"),
        LArgMain()  << EAM(toto,"toto",true,"Do no stuff")
    );

	//On créé un objet camera (aCam) et un objet liste de points d'appui (aSetGCP)
    cTD_Camera aCam(aNameCam);
    cTD_SetAppuis aSetGCP(aNameAppuis);

	//Pour chaque point d'appui, on calcul la distance entre la coordonnée image données par le fichier et la coordonnée image projetée à partir du point 3D et des infos de camera
    for (int aKP=0 ; aKP<int(aSetGCP.PTer().size()) ; aKP++)
    {
         Pt3dr aPTer = aSetGCP.PTer()[aKP];//Point 3D
         Pt2dr aPIm  = aSetGCP.PIm()[aKP];//Point image

         Pt2dr aPProj = aCam.Ter2Image(aPTer);//Point projeté

         std::cout  << "dist[" << aKP << "]= " << euclid (aPIm,aPProj) << "\n";
    }

	//On créé 3 int correspondant à des identifiants de points d'appui
    int aK1,aK2,aK3;
    std::cout << "ENTER K1 K2 K3 \n";
    cin >>  aK1 >> aK2 >>  aK3;
	//Avec ces 3 points, on calcule les positions et orientations possibles de la caméra
	
    std::vector<cTD_Camera> aSols = aCam.RelvtEspace
                                    (
                                          aSetGCP.PTer()[aK1], aSetGCP.PIm()[aK1],
                                          aSetGCP.PTer()[aK2], aSetGCP.PIm()[aK2],
                                          aSetGCP.PTer()[aK3], aSetGCP.PIm()[aK3]
                                    );

	//Pour chaque solution, on calcul la distance entre la coordonnée image données par le fichier et la coordonnée image projetée à partir du point 3D et des infos calculées
    for (int aKS=0 ; aKS<int(aSols.size()) ; aKS++)
    {
         for (int aKP=0 ; aKP<int(aSetGCP.PTer().size()) ; aKP++)
         {
              Pt3dr aPTer = aSetGCP.PTer()[aKP];
              Pt2dr aPIm  = aSetGCP.PIm()[aKP];

              Pt2dr aPProj = aSols[aKS].Ter2Image(aPTer);

              std::cout  << "   dist " << euclid (aPIm,aPProj) << "\n";
         }
         std::cout << "========================================\n";
    }

    return 0;
}


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/

double SimilariteAbsDif(cTD_Im aIm1,cTD_Im aIm2,int aX1,int aY, int aX2,int aSzW)
{
   double aDif = 0;
   for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
   {
      for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
      {
         aDif += fabs(aIm1.GetVal(aX1+aDx,aY+aDy)-aIm2.GetVal(aX2+aDx,aY+aDy));
      }
   }
   return aDif;
}

double SimilariteCorrel(cTD_Im aIm1,cTD_Im aIm2,int aX1,int aY, int aX2,int aSzW)
{
   double aS1=0.0;
   double aS2=0.0;
   double aS12=0.0;
   double aS11=0.0;
   double aS22=0.0;

   for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
   {
      for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
      {
          double aV1 = aIm1.GetVal(aX1+aDx,aY+aDy);
          double aV2 = aIm2.GetVal(aX2+aDx,aY+aDy);

          aS1 += aV1;
          aS2 += aV2;
          aS12 += aV1*aV2;
          aS11 += aV1*aV1;
          aS22 += aV2*aV2;
      }
   }

   double aSomP = (1+2*aSzW) *  (1+2*aSzW);

   aS1 /= aSomP;
   aS2 /= aSomP;
   aS12 /= aSomP;
   aS11 /= aSomP;
   aS22 /= aSomP;

   aS12 -= aS1 * aS2;
   aS11 -= aS1 * aS1;
   aS22 -= aS2 * aS2;

   double aCorrel = aS12 / sqrt(max(1e-10,aS11*aS22));

   return 1-aCorrel;
}


int TD_DiffImage(int argc,char ** argv)
{

    std::cout <<  "Bienvenu au TP MICMAC \n";
    std::string aNameIm1,aNameIm2;
    int         aSzW=2;
    int         aIntPx=100;
    bool        UseCorrel = true;
    std::string        Out = "TD-Disp";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of Image1")
                    << EAMC(aNameIm2,"Name of Image2"),
        LArgMain()  << EAM(aSzW,"SzW",true,"Size of window")
                    << EAM(aIntPx,"IntPx",true,"Intervalle of paralax")
                    << EAM(UseCorrel,"UseCor",true,"Use Correl instead of diff")
                    << EAM(Out,"Out",true,"Prefix for output")
    );

    cTD_Im aIm1 = cTD_Im::FromString(aNameIm1);
    cTD_Im aIm2 = cTD_Im::FromString(aNameIm2);


    int aTx1 = aIm1.Sz().x;
    int aTy =  min(aIm1.Sz().y,aIm2.Sz().y);
    int aTx2 = aIm2.Sz().x;

    cTD_Im aRes(aTx1,aTy);
    cTD_Im aSim(aTx1,aTy);

    for (int anY=aSzW ; anY<aTy-aSzW ; anY++)
    {
       for (int anX1=aSzW ; anX1<aTx1-aSzW ; anX1++)
       {
           int aXMin2 = max(aSzW,anX1-aIntPx);
           int aXMax2 = min(aTx2-aSzW,anX1+aIntPx+1);

           double aSimMin = 1e20;
           int anX2Opt=-100000;

           for (int anX2 = aXMin2 ; anX2<aXMax2  ; anX2++)
           {
               double aSim =  UseCorrel                                       ?
                             SimilariteCorrel(aIm1,aIm2,anX1,anY,anX2,aSzW)  :
                             SimilariteAbsDif(aIm1,aIm2,anX1,anY,anX2,aSzW)  ;
               if (aSim<aSimMin)
               {
                  aSimMin  = aSim;
                  anX2Opt = anX2;
               }
           }
           aRes.SetVal(anX1,anY,anX2Opt-anX1);
           aSim.SetVal(anX1,anY,aSimMin);
       }
       if ((anY%10)==0) std::cout << "Reste " << aTy - anY  << "\n";
    }

    aRes.Save(Out+"-Pax.tif");
    aSim.Save(Out+"-Sim.tif");

    return EXIT_SUCCESS;
}

void Normalize(cTD_Im & anIm)
{
	Pt2di aSz=anIm.Sz();
	
	double aVMax = -1e20;
	double aVMin = 1e20;
	
	for (int anX=0 ; anX<aSz.x ; anX++)
	{
		for (int anY=0 ; anY<aSz.y ; anY++)
		{
			double aVal = anIm.GetVal(anX,anY);
			aVMax = max(aVMax,aVal);
			aVMin = min(aVMin,aVal);

		}
	}
	
	for (int anX=0 ; anX<aSz.x ; anX++)
	{
		for (int anY=0 ; anY<aSz.y ; anY++)
		{
			double aVal = anIm.GetVal(anX,anY);
			aVal = (aVal-aVMin) / (aVMax-aVMin);
			anIm.SetVal(anX,anY,aVal);
		}
	}
}

int TD_CorrelQuick(int argc,char ** argv)
{

    std::cout <<  "Bienvenu au TP MICMAC \n";
    std::string aNameIm1,aNameIm2;
    int         aSzW=2;
    int         aIntPx=100;
    bool        UseCorrel = true;
    std::string        Out = "TD-Disp";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of Image1")
                    << EAMC(aNameIm2,"Name of Image2"),
        LArgMain()  << EAM(aSzW,"SzW",true,"Size of window")
                    << EAM(aIntPx,"IntPx",true,"Intervalle of paralax")
                    << EAM(UseCorrel,"UseCor",true,"Use Correl instead of diff")
                    << EAM(Out,"Out",true,"Prefix for output")
    );

    ElTimer aChrono;

    cTD_Im aIm1 = cTD_Im::FromString(aNameIm1);
    cTD_Im aIm2 = cTD_Im::FromString(aNameIm2);

 

    int aTx1 = aIm1.Sz().x;
    int aTy =  min(aIm1.Sz().y,aIm2.Sz().y);
    int aTx2 = aIm2.Sz().x;

    // [1] Calcul des images moyennes 
    cTD_Im  aMoy1 = aIm1.ImageMoy(aSzW,1);
    cTD_Im  aMoy2 = aIm2.ImageMoy(aSzW,1);


     // aMoy1.Save(Out+"MOY1.tif");
     // aIm1.Save(Out+"IM1.tif");

    // [2] Calcul des moyennes des carres 
    cTD_Im aIm11(aTx1,aTy);
    cTD_Im aIm22(aTx2,aTy);

         // [2.1] Calcule les carres
    for (int anY=0 ; anY<aTy ; anY++)
    {
         for (int aX1=0 ; aX1<aTx1 ; aX1++)
            aIm11.SetVal(aX1,anY,ElSquare(aIm1.GetVal(aX1,anY)));

         for (int aX2=0 ; aX2<aTx2 ; aX2++)
            aIm22.SetVal(aX2,anY,ElSquare(aIm2.GetVal(aX2,anY)));
    }
         // [2.2] Moyenne
    cTD_Im aMoy11 = aIm11.ImageMoy(aSzW,1);
    cTD_Im aMoy22 = aIm22.ImageMoy(aSzW,1);

         // [2.3] Moyenne
    for (int anY=0 ; anY<aTy ; anY++)
    {
         for (int aX1=0 ; aX1<aTx1 ; aX1++)
         {

            aMoy11.SetVal(aX1,anY,aMoy11.GetVal(aX1,anY)-ElSquare(aMoy1.GetVal(aX1,anY)));
         }

         for (int aX2=0 ; aX2<aTx2 ; aX2++)
            aMoy22.SetVal(aX2,anY,aMoy22.GetVal(aX2,anY)-ElSquare(aMoy2.GetVal(aX2,anY)));
    }


    // [3] Calcul des paralaxes

         // [3.1] => Initialise Image de Pax et Image de similiarite
    cTD_Im aPaxOpt(aTx1,aTy);
    cTD_Im aImSim(aTx1,aTy);

    for (int anY=0 ; anY<aTy ; anY++)
         for (int aX1=0 ; aX1<aTx1 ; aX1++)
         {
             aPaxOpt.SetVal(aX1,anY,aIntPx+1);  // Valeur en dehors du domaine
             aImSim.SetVal(aX1,anY,1e20);         // + l'infini
         }

    for (int aPax=-aIntPx ; aPax <= aIntPx ;aPax++)
    {
        // calclul dans aMoy12 de moyenne de  I1*( I2 decalee )
        cTD_Im aMoy12(aTx1,aTy);  
        for (int anY=0 ; anY<aTy ; anY++)
        {
            for (int aX1=0 ; aX1<aTx1 ; aX1++)
            {
                int aX2 = aX1+aPax;
                if (aIm2.Ok(aX2,anY))
                   aMoy12.SetVal(aX1,anY,aIm1.GetVal(aX1,anY)*aIm2.GetVal(aX2,anY));
                else
                   aMoy12.SetVal(aX1,anY,0);
            }
        }
        aMoy12 = aMoy12.ImageMoy(aSzW,1);

       
        for (int anY=0 ; anY<aTy ; anY++)
        {
            for (int aX1=0 ; aX1<aTx1 ; aX1++)
            {
                int aX2 = aX1+aPax;
                if (aIm2.Ok(aX2,anY))
                {
                     double aV1  = aMoy1.GetVal(aX1,anY);
                     double aV11 = aMoy11.GetVal(aX1,anY);
                     double aV2  = aMoy2.GetVal(aX2,anY);
                     double aV22 = aMoy22.GetVal(aX2,anY);
                     double aV12 = aMoy12.GetVal(aX1,anY) - aV1 * aV2;
                     double aCorrel = aV12 / sqrt(max(aV11*aV22,double(1e-5)));

if ((aCorrel > 10) || (aCorrel <-10))
{
	std::cout 
	     << " aV1 " << aV1
	       << " aV2 " << aV2
	         << " aV11 " << aV11
	           << " aV22 " << aV22
	             << " aV12 " << aV12
	             << "XY=" << aX1 << " " << anY << " X2 " << aX2
	             << "\n";
	            //getchar();
}

                     if ((aCorrel<=1) && (aCorrel>=-1))
                     {

                         double aSim = 1- aCorrel;

                         if (aSim < aImSim.GetVal(aX1,anY))
                         {
                              aImSim.SetVal(aX1,anY,aSim);
                             aPaxOpt.SetVal(aX1,anY,aPax);
                          }
				     }

                }
                else
                {
                }
            }
        }


    }


    aPaxOpt.Save(Out+"-Pax.tif");
    aImSim.Save(Out+"-Sim.tif");
    
    std::cout << " TIME " << aChrono.uval() << "\n";

    return EXIT_SUCCESS;
}


//================================================================

class cAppli_TD_DBayer
{
    public :
        cAppli_TD_DBayer(int argc,char ** argv);

        cTD_Im  ImWith0(const std::string & aFilter);
        cTD_Im  BayerConvol(cTD_Im,float aDiv);

        std::string mNameImIn;
        cTD_Im      mIn;
        Pt2di       mSzIm;
};

cTD_Im cAppli_TD_DBayer::BayerConvol(cTD_Im aImIn,float aDiv)
{
     cTD_Im aRes(mSzIm.x,mSzIm.y);
     
     Pt2di aP;
     for (aP.x=1 ; aP.x<mSzIm.x -1  ; aP.x++)
     {
         for (aP.y=1 ; aP.y<mSzIm.y-1 ; aP.y++)
         {
               float aVal = 
                               1 * aImIn.GetVal(aP.x-1,aP.y-1)
                             + 2 * aImIn.GetVal(aP.x  ,aP.y-1)
                             + 1 * aImIn.GetVal(aP.x+1,aP.y-1)
                             + 2 * aImIn.GetVal(aP.x-1,aP.y  )
                             + 4 * aImIn.GetVal(aP.x  ,aP.y  )
                             + 2 * aImIn.GetVal(aP.x+1,aP.y  )
                             + 1 * aImIn.GetVal(aP.x-1,aP.y+1)
                             + 2 * aImIn.GetVal(aP.x  ,aP.y+1)
                             + 1 * aImIn.GetVal(aP.x+1,aP.y+1) ;

                  aVal = aVal /16.0;
                  aRes.SetVal(aP.x,aP.y,aVal/aDiv);
         }
    }
    return aRes;
}

cTD_Im cAppli_TD_DBayer::ImWith0(const std::string & aFilter)
{
     std::cout << "DO PATTERN " <<  aFilter << "\n";
     cTD_Im aRes(mSzIm.x,mSzIm.y);

     Pt2di aP;
     for (aP.x=0 ; aP.x<mSzIm.x ; aP.x++)
     {
         for (aP.y=0 ; aP.y<mSzIm.y ; aP.y++)
         {
              float aVal = 0;
              int aK = (aP.x%2) + 2* (aP.y%2);
              if (aFilter[aK]=='0')
              {
              }
              else if (aFilter[aK]=='+')
              {
                  aVal = mIn.GetVal(aP);
              }
              else 
              {
                   ELISE_ASSERT(false,"Bad carac");
              }
              aRes.SetVal(aP.x,aP.y,aVal);
         }
     }

     return aRes;
}


cAppli_TD_DBayer::cAppli_TD_DBayer(int argc,char ** argv):
   mIn(1,1)
{
      
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameImIn,"Name of Image1"),
        LArgMain()
    );

    mIn = cTD_Im::FromString(mNameImIn);
    mSzIm = mIn.Sz();

    cTD_Im aImR = ImWith0("+000");  aImR.Save("Rouge-0.tif");
    aImR = BayerConvol(aImR,1.0);  aImR.Save("RougePlein.tif");

    cTD_Im aImV = ImWith0("0++0");  aImV.Save("Vert-0.tif");
    aImV = BayerConvol(aImV,2.0);  aImV.Save("VertPlein.tif");

    cTD_Im aImB = ImWith0("000+");  aImB.Save("Bleu-0.tif");
    aImB = BayerConvol(aImB,1.0);  aImB.Save("BleuPlein.tif");


    aImR.SaveRGB(std::string("DeBayer-"+mNameImIn),aImV,aImB);
}

class cAppli_TD_RhoTetaBayer
{
     public :
         cAppli_TD_RhoTetaBayer(int argc,char ** argv);
     private :
        std::string mNameImX;
        std::string mNameImY;
        cTD_Im      mDx;
        cTD_Im      mDy;
        cTD_Im      mImRho;
        cTD_Im      mImTeta;
        Pt2di       mSz;
        Pt2dr       mDecal;
};

cAppli_TD_RhoTetaBayer::cAppli_TD_RhoTetaBayer(int argc,char ** argv) :
     mDx     (1,1),
     mDy     (1,1),
     mImRho  (1,1),
     mImTeta (1,1),
     mDecal  (0.0,0.0)
{
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameImX,"Name of Dep X")
                    << EAMC(mNameImY,"Name of Dep Y") ,

        LArgMain()  << EAM(mDecal,"Decal",true,"Decalage central")
    );
    mDx = cTD_Im::FromString(mNameImX);
    mDy = cTD_Im::FromString(mNameImY);
    mSz = mDx.Sz();

    mImRho  =  cTD_Im(mSz.x,mSz.y);
    mImTeta =  cTD_Im(mSz.x,mSz.y);

    Pt2di aP;
    for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
            Pt2dr aQ (mDx.GetVal(aP),mDy.GetVal(aP));
            aQ = aQ-mDecal;
            aQ  = Pt2dr::polar(aQ,0.0);
            mImRho.SetVal(aP.x,aP.y,aQ.x);
            mImTeta.SetVal(aP.x,aP.y,aQ.y);
          
        }
    }
     mImRho.Save("Rho.tif");
    mImTeta.Save("Teta.tif");
}


int TD_Exemple_main(int argc,char ** argv)
{
    cAppli_TD_RhoTetaBayer(argc,argv);
    // return TD_CorrelQuick(argc,argv);
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
