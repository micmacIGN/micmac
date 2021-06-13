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
#if (0)

// mm3d ElDcraw -t 0  -d -T -4   IMG_1364-CL1.CR2 

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/

int TD_Exo0(int argc,char ** argv)
{
    std::string aNameIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name of image"),
        LArgMain()  
    );

    cTD_Im aIm = cTD_Im::FromString(aNameIm);

    std::cout << "Bonjour , NameIm=" << aNameIm  
              << " Sz=" <<  aIm.Sz().x 
              << " : " <<  aIm.Sz().y << "\n";

    cTD_Im aImRes(aIm.Sz().x,aIm.Sz().y);

    double aVMax = (1<<16) -1;
    for (int aX=0 ; aX<aIm.Sz().x ; aX++)
    {
        for (int aY=0 ; aY<aIm.Sz().y ; aY++)
        {
            double aVal = aIm.GetVal(aX,aY);
            aImRes.SetVal(aX,aY,aVMax-aVal);
        }
    }
    aImRes.Save("Neg.tif");


    return EXIT_SUCCESS;
}


int TD_Exo2(int argc,char ** argv)
{
    std::string aNameDepX,aNameDepY,aNameIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameDepX,"Deplacement X")
                    << EAMC(aNameDepY,"Deplacement Y")
                    << EAMC(aNameIm,"Image to correct"),
        LArgMain()  
    );

    cTD_Im anImDepX = cTD_Im::FromString(aNameDepX);
    cTD_Im anImDepY = cTD_Im::FromString(aNameDepY);
    cTD_Im anImIn = cTD_Im::FromString(aNameIm);

    Pt2di aSz = anImDepX.Sz();
    double aMilX = aSz.x/2.0;
    double aMilY = aSz.y/2.0;

    std::vector<double> aVLx;
    std::vector<double> aVLy;

    for (int aX=0 ; aX<aSz.x; aX++)
    {
        for (int aY=0 ; aY<aSz.y; aY++)
        {
             double aXTheo = aX-aMilX;
             double aYTheo = aY-aMilY;
             double aXDepl = anImDepX.GetVal(aX,aY);
             double aYDepl = anImDepY.GetVal(aX,aY);
             if (aXTheo !=0)
                aVLx.push_back(aXDepl/aXTheo);
             if (aYTheo !=0)
                aVLy.push_back(aYDepl/aYTheo);
        }
    }
    std::sort(aVLx.begin(),aVLx.end());
    std::sort(aVLy.begin(),aVLy.end());

    double aLx = aVLx.at(aVLx.size()/2);
    double aLy = aVLy.at(aVLy.size()/2);

    std::cout << "Lx=" << aLx << " Ly=" << aLy << "\n";

    double aLambda = (aLx+aLy) / 2.0;
  
    cTD_Im aImResidu(aSz.x,aSz.y);
    cTD_Im aImOut(aSz.x,aSz.y);
    double aSomResidu = 0.0;
    double aNbPix = 0.0;
    
    for (int aX=0 ; aX<aSz.x; aX++)
    {
        for (int aY=0 ; aY<aSz.y; aY++)
        {
             double aXTheo = (aX-aMilX)*aLambda;
             double aYTheo = (aY-aMilY)*aLambda;
             double aXDepl = anImDepX.GetVal(aX,aY);
             double aYDepl = anImDepY.GetVal(aX,aY);
             
             double aDx = aXTheo-aXDepl;
             double aDy = aYTheo-aYDepl;
             double aDif = sqrt(aDx*aDx+aDy*aDy);
             aImResidu.SetVal(aX,aY,aDif);
             aSomResidu += aDif;
             aNbPix++;
             
             Pt2dr aPCorr(aX-aXTheo,aY-aYTheo);
             aImOut.SetVal(aX,aY,anImIn.GetVal(aPCorr));
             
        }
    }   
    std::cout << "RESIDU MOY " << aSomResidu / aNbPix << "\n";
    aImResidu.Save("Residu.tif");
    aImOut.Save("Deform-"+aNameIm);
    
    return EXIT_SUCCESS;
}



int TD_Exo1(int argc,char ** argv)
{
    std::string aNameIm;
    Pt2dr aRatioRBdV(1.0,1.0);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name of image"),
        LArgMain()  
                    << EAM(aRatioRBdV,"RBdV",true,"Experimental ratios R/V and R/B")
    );

    // Lecture de la matrice de Bayer
    cTD_Im aIm = cTD_Im::FromString(aNameIm);

    // calcul de la taille des 4 matrice R,V1,V2,B

    int aSzRedX = aIm.Sz().x/2 ;
    int aSzRedY = aIm.Sz().y/2 ;

    // Creation de 4 image reduite : allocation memoire
    // contenu non initialise
    cTD_Im aImR(aSzRedX,aSzRedY);
    cTD_Im aImV(aSzRedX,aSzRedY);
    cTD_Im aImW(aSzRedX,aSzRedY);
    cTD_Im aImB(aSzRedX,aSzRedY);

    cTD_Im aImRdV(aSzRedX,aSzRedY);
    cTD_Im aImBdV(aSzRedX,aSzRedY);
    cTD_Im aImVdV(aSzRedX,aSzRedY);

    // Boucle pour remplir les valeur
    for (int aX=0 ; aX<aSzRedX ; aX++)
    {
        for (int aY=0 ; aY<aSzRedY ; aY++)
        {
            // les pixels rouges correspondent aux coordonnees
            // paires en x et y
            double aV1  = aIm.GetVal(2*aX  ,2*aY  );
            double aV2  = aIm.GetVal(2*aX+1,2*aY+1  );
            double aR   = aIm.GetVal(2*aX+1,2*aY  ) / aRatioRBdV.x;
            double aB   = aIm.GetVal(2*aX  ,2*aY+1  )/aRatioRBdV.y;
            aImR.SetVal(aX,aY,aR);
            aImV.SetVal(aX,aY,aV1);
            aImW.SetVal(aX,aY,aV2);
            aImB.SetVal(aX,aY,aB);

            // std::max => evite la division par 0
            aImRdV.SetVal(aX,aY,aR/std::max(1.0,aV1));
            aImBdV.SetVal(aX,aY,aB/std::max(1.0,aV1));
            aImVdV.SetVal(aX,aY,aV2/std::max(1.0,aV1));
        }
    }
    
    cTD_Im aImRCorr(aSzRedX,aSzRedY);
    cTD_Im aImWCorr(aSzRedX,aSzRedY);
    cTD_Im aImBCorr(aSzRedX,aSzRedY);
    
    cTD_Im aImVdWCorr(aSzRedX,aSzRedY);
    for (int aX=0 ; aX<aSzRedX ; aX++)
    {
        for (int aY=0 ; aY<aSzRedY ; aY++)
        {
			double aV = aImV.GetVal(aX,aY);
			double aWCor = aImW.GetVal(Pt2dr(aX-0.5,aY-0.5));
			double aRCor = aImR.GetVal(Pt2dr(aX-0.5,aY    ));
			double aBCor = aImB.GetVal(Pt2dr(aX    ,aY-0.5));
			
			aImRCorr.SetVal(aX,aY,aRCor);
			aImWCorr.SetVal(aX,aY,aWCor);
			aImBCorr.SetVal(aX,aY,aBCor);
			aImVdWCorr.SetVal(aX,aY,aWCor/std::max(1.0,aV));
		}
	}

    // Sauvegarde dans 4 fichiers separes
    aImR.Save("ROUGE-"+aNameIm+".tif");
    aImV.Save("VERT1-"+aNameIm+".tif");
    aImW.Save("VERT2-"+aNameIm+".tif");
    aImB.Save("BLEU-" +aNameIm+".tif");

    aImRdV.Save("RdV-" +aNameIm+".tif");
    aImBdV.Save("BdV-" +aNameIm+".tif");
    aImVdV.Save("VdV-" +aNameIm+".tif");
    aImVdWCorr.Save("VCordV-"+aNameIm+".tif");

    aImR.SaveRGB("RGB-"+aNameIm+".tif",aImV,aImB);
    
    aImRCorr.SaveRGB("RGB-Corr-"+aNameIm+".tif",aImV,aImBCorr);
    aImRCorr.Save("ROUGE-Corr"+aNameIm+".tif");
    aImBCorr.Save("BLEU-Corr" +aNameIm+".tif");

    return EXIT_SUCCESS;
}
 enum eTypeCorrel
{
	eScoreL1,
	eScoreCorrel,
	eScoreCensus
};

class cResCorrel
{
    public :
       cTD_Im mImPx;
       cTD_Im mImCorrel;

       cResCorrel  (Pt2di aSz) :
           mImPx     (aSz.x,aSz.y),
           mImCorrel (aSz.x,aSz.y)
       {
       }
};

class cTD_CorrelEpip
{
    public :
       cTD_CorrelEpip(const cTD_Im & anI1n,const cTD_Im & anI2,eTypeCorrel);

       float Score(int aX1,int aY,int aX2,Pt2di aSzW);
       
       
       float ScoreL1(int aX1,int aY,int aX2,Pt2di aSzW);
       float ScoreCorrel(int aX1,int aY,int aX2,Pt2di aSzW);
       float ScoreCensus(int aX1,int aY,int aX2,Pt2di aSzW);


       cResCorrel  DoCorrel(int aDPx,Pt2di aSzW);
    protected :
       cTD_Im  mI1;
       cTD_Im  mI2;
       Pt2di   mSz1;
       Pt2di   mSz2;
       eTypeCorrel mType;
};


cTD_CorrelEpip::cTD_CorrelEpip(const cTD_Im & anI1,const cTD_Im & anI2,eTypeCorrel aType) :
   mI1  (anI1),
   mI2  (anI2),
   mSz1 (anI1.Sz()),
   mSz2 (anI1.Sz()),
   mType (aType)
{
}

float cTD_CorrelEpip::ScoreL1(int aX1,int aY,int aX2,Pt2di aSzW)
{
    float aSomDif = 0.0;
    for (int aDx=-aSzW.x ;  aDx <= aSzW.x ; aDx++)
    {
        for (int aDy=-aSzW.y ;  aDy <= aSzW.y ; aDy++)
        {
              float aV1  = mI1.GetVal(aX1+aDx,aY+aDy);
              float aV2  = mI2.GetVal(aX2+aDx,aY+aDy);
              aSomDif += fabs(aV1-aV2);
        }
    }
    return aSomDif;
}

float cTD_CorrelEpip::ScoreCensus(int aX1,int aY,int aX2,Pt2di aSzW)
{

    int aNb = (1+2*aSzW.x) * (1+2*aSzW.y);
    float aVC1 = mI1.GetVal(aX1,aY);
    float aVC2 = mI2.GetVal(aX2,aY);
    int aNbDiff = 0;
    
    for (int aDx=-aSzW.x ;  aDx <= aSzW.x ; aDx++)
    {
        for (int aDy=-aSzW.y ;  aDy <= aSzW.y ; aDy++)
        {
              float aV1  = mI1.GetVal(aX1+aDx,aY+aDy);
              float aV2  = mI2.GetVal(aX2+aDx,aY+aDy);
              aNbDiff +=  (aV1>aVC1) != (aV2>aVC2);
        }
    }
    // std::cout <<  aNbDiff / double(aNb) << "\n";
    return aNbDiff / double(aNb);
}


float cTD_CorrelEpip::ScoreCorrel(int aX1,int aY,int aX2,Pt2di aSzW)
{
    double aS1  = 0.0;
    double aS2  = 0.0;
    double aS11 = 0.0;
    double aS22 = 0.0;
    double aS12 = 0.0;

    int aNb = (1+2*aSzW.x) * (1+2*aSzW.y);
    
    for (int aDx=-aSzW.x ;  aDx <= aSzW.x ; aDx++)
    {
        for (int aDy=-aSzW.y ;  aDy <= aSzW.y ; aDy++)
        {
              float aV1  = mI1.GetVal(aX1+aDx,aY+aDy);
              float aV2  = mI2.GetVal(aX2+aDx,aY+aDy);
              aS1 += aV1;
              aS2 += aV2;
              aS11 += aV1*aV1;
              aS12 += aV1*aV2;
              aS22 += aV2*aV2;
        }
    }

    aS1 /= aNb;
    aS2 /= aNb;
    aS11 = aS11/aNb - aS1*aS1;
    aS12 = aS12/aNb - aS1*aS2;
    aS22 = aS22/aNb - aS2*aS2;

    double aCorr = aS12 / sqrt(ElMax(1e-5,aS11*aS22));

    return 1-aCorr;
}


float cTD_CorrelEpip::Score(int aX1,int aY,int aX2,Pt2di aSzW)
{

	switch (mType)
	{
		case eScoreL1 : return ScoreL1(aX1,aY,aX2,aSzW);
		case eScoreCorrel : return ScoreCorrel(aX1,aY,aX2,aSzW);
		case eScoreCensus : return ScoreCensus(aX1,aY,aX2,aSzW);
		
		default : ;
	}
	
	return 0;
}

cResCorrel cTD_CorrelEpip::DoCorrel(int aDeltaPax,Pt2di aSzW)
{
    cResCorrel aRes(mSz1);
    Pt2di aP;
    for (aP.x=aSzW.x ; aP.x<mSz1.x-aSzW.x ; aP.x++)
    {
        int aX2Min =  ElMax(aSzW.x,aP.x-aDeltaPax);
        int aX2Max =  ElMin(mSz2.x-aSzW.x,aP.x+aDeltaPax);
        for (aP.y=aSzW.y ; aP.y<mSz1.y-aSzW.y ; aP.y++)
        {
             float aDifMin = 1e20;
             int aPaxMin = 0;
             for (int aX2 = aX2Min ; aX2 < aX2Max; aX2++)
             {
                 int aPx = aX2-aP.x;
                 float aSomDif = Score(aP.x,aP.y,aX2,aSzW);
                 if (aSomDif<aDifMin)
                 {
                    aDifMin = aSomDif;
                    aPaxMin = aPx;
                 }
             }
             aRes.mImPx.SetVal(aP.x,aP.y,aPaxMin);
             aRes.mImCorrel.SetVal(aP.x,aP.y,aDifMin);
// std::cout << "GGGG " <<  ScoreCorrel(aP.x,aP.y,aPaxMin+aP.x,aSzW) << " " << ScoreCensus(aP.x,aP.y,aPaxMin+aP.x,aSzW) << "\n";
        }
    }
    return aRes;
}






cTD_Im EcartPx(cTD_Im aPx12, cTD_Im aPx21)
{
   Pt2di aSz1 = aPx12.Sz();
   cTD_Im aRes(aSz1.x,aSz1.y);

   Pt2di aP;
   for (aP.y=0 ; aP.y<aSz1.y ; aP.y++)
   {
       for (aP.x=0 ; aP.x<aSz1.x ; aP.x++)
       {
           int aPx1 = round_ni(aPx12.GetVal(aP));
           int aX2  = aP.x+aPx1;
           if (aPx21.Ok(aX2,aP.y))
           {
               int aPx2 = aPx21.GetVal(aX2,aP.y);
               aRes.SetVal(aP.x,aP.y,ElAbs(aPx1+aPx2));
           }
           else
           {
               aRes.SetVal(aP.x,aP.y,100);
           }
       }
   }

   return aRes;
}

int TD_Exo3(int argc,char ** argv)
{
    std::string aNameIm1,aNameIm2;
    Pt2di aSzW(2,2);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image"),
        LArgMain()  << EAM(aSzW,"SzW",true,"Sz Vignette")
    );

    cTD_Im anImIn1 = cTD_Im::FromString(aNameIm1);
    cTD_Im anImIn2 = cTD_Im::FromString(aNameIm2);
    // eTypeCorrel aType = eScoreCensus;
    eTypeCorrel aType = eScoreCorrel;

    std::cout << "BEGIN \n";

    cTD_CorrelEpip aCorrel12(anImIn1,anImIn2,aType);
    cResCorrel  aRC12 = aCorrel12.DoCorrel(100,aSzW);
    cTD_Im aPx12 = aRC12.mImPx;
    std::cout << "Done One Way \n";

    cTD_CorrelEpip aCorrel21(anImIn2,anImIn1,aType);
    cResCorrel  aRC21 = aCorrel21.DoCorrel(100,aSzW);
    cTD_Im aPx21 = aRC21.mImPx;
    std::cout << "Done Two Way \n";


    cTD_Im anEcart12 = EcartPx(aPx12,aPx21);
    cTD_Im anEcart21 = EcartPx(aPx21,aPx12);

    
    anEcart12.Save("Ec12.tif");
    anEcart21.Save("Ec21.tif");

    return EXIT_SUCCESS;
}

/****************************************************/
/*                                                  */
/*            cTD_QuickCorEpip                      */
/*                                                  */
/****************************************************/

// Calcul l'image produit de deuc images

cTD_Im MulImage(const cTD_Im & anI1,const cTD_Im & anI2)
{
   Pt2di aSz1 = anI1.Sz();
   // On ne veut pas gerer le cas ou les images auraient une taille differente
   ELISE_ASSERT(aSz1==anI2.Sz(),"Dif size in MulImage");

   cTD_Im aRes(aSz1.x,aSz1.y);
   Pt2di aP;

   for (aP.y=0 ; aP.y<aSz1.y ; aP.y++)
   {
       for (aP.x=0 ; aP.x<aSz1.x ; aP.x++)
       {
           aRes.SetVal
           (
              aP.x, aP.y,
              anI1.GetVal(aP.x,aP.y) * anI2.GetVal(aP.x,aP.y)
           );
       }
   }
   return aRes;
}

class cTD_QuickCorEpip : public  cTD_CorrelEpip
{
    public :
       cTD_QuickCorEpip(const cTD_Im & anI1n,const cTD_Im & anI2,int aSzW,int aNbIter=1);
       
       void  DoAllPx(int aPax);  // Fait le calcul sur toute les paralaxe
       cTD_Im & ImPx() {return mImPx;}
       cTD_Im & ImCor() {return mImCor;}

    private :

       void DoOnePx(int aPax);  // Effectue le calcul pour une paralaxe donnee

       int     mSzW;
       int     mNbIter;
       cTD_Im  mM1;   // Moyenne de I1
       cTD_Im  mM2;   // Moyenne de I2
       cTD_Im  mM11;   // Moyenne de I1 au carre
       cTD_Im  mM22;   // Moyenne de I2 au carre
       cTD_Im  mI12;   // Produit e I1 I2 translate
       cTD_Im  mM12;   // Moyenne I12
       cTD_Im  mImPx;  // Paralaxe donnant la meilleur correlation
       cTD_Im  mImCor; //  Meilleure correlation obtenue
};

cTD_QuickCorEpip::cTD_QuickCorEpip(const cTD_Im & anI1,const cTD_Im & anI2,int aSzW,int aNbIter) :
   cTD_CorrelEpip (anI1,anI2,eScoreCorrel),
   mSzW           (aSzW),
   mNbIter        (aNbIter),
   mM1            (mI1.ImageMoy(mSzW,mNbIter)),
   mM2            (mI2.ImageMoy(mSzW,mNbIter)),
   mM11           (MulImage(mI1,mI1).ImageMoy(mSzW,mNbIter)),
   mM22           (MulImage(mI2,mI2).ImageMoy(mSzW,mNbIter)),
   mI12           (mSz1.x,mSz1.y),
   mM12           (mSz1.x,mSz1.y),
   mImPx          (mSz1.x,mSz1.y),
   mImCor         (mSz1.x,mSz1.y)
{
   // Initialisation des score de correl a "+ infini"
   Pt2di aP;
   for (aP.y=0 ; aP.y<mSz1.y ; aP.y++)
   {
       for (aP.x=0 ; aP.x<mSz1.x ; aP.x++)
       {
           mImCor.SetVal(aP.x,aP.y,1e10);
           mImPx.SetVal(aP.x,aP.y,0);
       }
   }
}

void  cTD_QuickCorEpip::DoAllPx(int aPaxMax)
{
    for (int aPax = -aPaxMax  ; aPax<=aPaxMax ; aPax++)
    {
        std::cout << "cTD_QuickCorEpip::DoAllPx " << aPax << "\n";
        DoOnePx(aPax);
    }
}

void cTD_QuickCorEpip::DoOnePx(int aPax)
{
   // Calcul de I1 * Trans(I2) 
   Pt2di aP;
   for (aP.y=0 ; aP.y<mSz1.y ; aP.y++)
   {
       for (aP.x=0 ; aP.x<mSz1.x ; aP.x++)
       {
           int aX2 = aP.x + aPax;
           if (mI2.Ok(aX2,aP.y))
               mI12.SetVal(aP.x,aP.y,mI1.GetVal(aP.x,aP.y)*mI2.GetVal(aX2,aP.y));
           else
               mI12.SetVal(aP.x,aP.y,0);
       }
   }
   // Calcul de la moyenne mI12 dans mM12
   mM12 = mI12.ImageMoy(mSzW,mNbIter);


   for (aP.y=0 ; aP.y<mSz1.y ; aP.y++)
   {
       for (aP.x=0 ; aP.x<mSz1.x ; aP.x++)
       {
           int aX2 = aP.x + aPax;
           if (mI2.Ok(aX2,aP.y))
           {
               // Calcul du coefficient de correlation selon la formule habituelle
               float aS1  = mM1.GetVal(aP.x,aP.y);
               float aS2  = mM2.GetVal(aX2,aP.y);
               float aS11 = mM11.GetVal(aP.x,aP.y)- aS1*aS1;
               float aS22 = mM22.GetVal(aX2,aP.y) - aS2*aS2;
               float aS12 = mM12.GetVal(aP.x,aP.y) -aS1*aS2;

               double  aCor = 1-aS12 / sqrt(ElMax(1e-5,double(aS11*aS22)));

               // Si meilleure que la valeur courant, on met a jour
               if (aCor < mImCor.GetVal(aP.x,aP.y))
               {
                   mImCor.SetVal(aP.x,aP.y,aCor);
                   mImPx.SetVal(aP.x,aP.y,aPax);
               }
           }
       }
   }
}

int TD_Exo4(int argc,char ** argv)
{
    std::string aNameIm1,aNameIm2;
    int  aSzW = 2;
    int  aIntPx = 100;
    int  aNbIter = 1;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image"),
        LArgMain()  << EAM(aSzW,"SzW",true,"Sz Vignette")
                    << EAM(aIntPx,"IntPx",true,"Interval Pax, Def=100")
                    << EAM(aNbIter,"NbIter",true,"Nombre iteration, Def=1")
    );

    cTD_Im anImIn1 = cTD_Im::FromString(aNameIm1);
    cTD_Im anImIn2 = cTD_Im::FromString(aNameIm2);


    cTD_QuickCorEpip aQC(anImIn1,anImIn2,aSzW,aNbIter);

    aQC.DoAllPx(aIntPx);

    aQC.ImPx().Save("QkPx.tif");

    return EXIT_SUCCESS;
}

int TD_Exo5(int argc,char ** argv)
{
    std::string aNameImBay;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameImBay,"Name of image"),
        LArgMain()  
    );

    cTD_Im anImIn1 = cTD_Im::FromString(aNameImBay);

    Pt2di aSzF = anImIn1.Sz();
    Pt2di aSz2 = aSzF / 2;
    cTD_Im anImInR(aSz2.x,aSz2.y);
    cTD_Im anImInB(aSz2.x,aSz2.y);
    cTD_Im anImInV(aSz2.x,aSz2.y);
    cTD_Im anImInW(aSz2.x,aSz2.y);

    for (int aX=0 ; aX<aSz2.x ; aX++)
    {
        for (int aY=0 ; aY<aSz2.y ; aY++)
        {
              anImInR.SetVal(aX,aY,anImIn1.GetVal(aX*2,aY*2));
              anImInB.SetVal(aX,aY,anImIn1.GetVal(aX*2+1,aY*2+1));
              anImInV.SetVal(aX,aY,anImIn1.GetVal(aX*2,aY*2+1));
              anImInW.SetVal(aX,aY,anImIn1.GetVal(aX*2+1,aY*2));
        }
    }
    anImInR.Save("R-"+StdPrefix(aNameImBay)+".tif");
    anImInB.Save("B-"+StdPrefix(aNameImBay)+".tif");
    anImInV.Save("V-"+StdPrefix(aNameImBay)+".tif");
    anImInW.Save("W-"+StdPrefix(aNameImBay)+".tif");
    return EXIT_SUCCESS;
}

int CorrelRapide_TD_Exo6(int argc,char ** argv)
{
    std::string aNameImBay;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameImBay,"Name of image"),
        LArgMain()  
    );

    cTD_Im anImIn1 = cTD_Im::FromString(aNameImBay);
    Pt2di aSzF = anImIn1.Sz();

    cTD_Im anImInR(aSzF.x,aSzF.y);
    cTD_Im anImInV(aSzF.x,aSzF.y);
    cTD_Im anImInB(aSzF.x,aSzF.y);


    for (int aX=0 ; aX<aSzF.x ; aX++)
    {
        // std::cout << aX << "\n";
        for (int aY=0 ; aY<aSzF.y ; aY++)
        {
             double aSomPdsR=0;
             double aSomPdsV=0;
             double aSomPdsB=0;
             double aSomR=0;
             double aSomV=0;
             double aSomB=0;

             for (int dX=-1 ; dX<=1 ; dX++)
             {
                 for (int dY=-1 ; dY<=1 ; dY++)
                 {
                      int aXV = aX + dX;
                      int aYV = aY + dY;
                      if (anImIn1.Ok(aXV,aYV))
                      {
                          double aPds = (2-fabs((double)dX) ) * (2-fabs((double)dY));
                          float aValP = anImIn1.GetVal(aXV,aYV) * aPds;
                          if ((aXV%2) != (aYV%2))
                          {
                            aSomPdsV += aPds;
                            aSomV += aValP;
                          }
                          else 
                          {
                              if (aXV%2==0) 
                              {
                                 aSomPdsR += aPds;
                                 aSomR += aValP;
                              }
                              else
                              {
                                 aSomPdsB += aPds;
                                 aSomB += aValP;
                              }
                          }
                      }
                 }
             }
             anImInR.SetVal(aX,aY,aSomR/aSomPdsR);
             anImInV.SetVal(aX,aY,aSomV/aSomPdsV);
             anImInB.SetVal(aX,aY,aSomB/aSomPdsB);
        }
    }
    anImInR.SaveRGB("Debay-"+ StdPrefix(aNameImBay) +".tif",anImInV,anImInB);
    return EXIT_SUCCESS;
}

// Test Decal
int TD_Exo6(int argc,char ** argv)
{
    std::string aNameIm1;
    std::string aNameIm2;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image"),
        LArgMain()  
    );

    double aStep = 0.25;
    double aMaxDep = 1.0;
    int aSzW = 2;

    int aNbDec = ceil(aMaxDep / aStep);

    cTD_Im anImIn1 = cTD_Im::FromString(aNameIm1);
    cTD_Im anImIn2 = cTD_Im::FromString(aNameIm2);
   
    Pt2di aSzF = anImIn1.Sz();

    cTD_Im anImDecX(aSzF.x,aSzF.y);
    cTD_Im anImDecY(aSzF.x,aSzF.y);

    int aBrd = 0;
    // Parcour des XY de l'image 1
    for (int aX = aBrd ; aX < aSzF.x -aBrd ; aX++)
    {
        std::cout << "RESTE " << aSzF.x -aX << " lignes \n";
        for (int aY = aBrd ; aY < aSzF.y -aBrd ; aY++)
        {
             double aCorMax = -1e10;
             Pt2dr  aDecMax(0,0);
             Pt2di aC1(aX,aY); // Centre Vignette 1
             // Parcour des decalage
             for (int aDx2 = -aNbDec ; aDx2<=aNbDec ; aDx2++)
             {
                 for (int aDy2 = -aNbDec ; aDy2<=aNbDec ; aDy2++)
                 {
                     Pt2dr aDec (aDx2*aStep, aDy2 *aStep);
                     Pt2dr aC2 = Pt2dr(aC1) + aDec;
                     double aS1 = 0;
                     double aS2 = 0;
                     double aS11 = 0;
                     double aS12 = 0;
                     double aS22 = 0;
                     int aNbPix = (1+2*aSzW) *  (1+2*aSzW);
                     
                     // Parcour de la vignette
                     for (int aWx=-aSzW ; aWx<=aSzW ; aWx++)
                     {
                         for (int aWy=-aSzW ; aWy<=aSzW ; aWy++)
                         {
                             double aV1 = anImIn1.GetVal(Pt2dr(aC1+Pt2di(aWx,aWy)));
                             double aV2 = anImIn2.GetVal(aC2+Pt2dr(aWx,aWy));

                             aS1 += aV1;
                             aS2 += aV2;
                             aS11 += aV1 * aV1;
                             aS12 += aV1 * aV2;
                             aS22 += aV2 * aV2;
                         }
                     }
                     aS1 /= aNbPix;
                     aS2 /= aNbPix;
                     aS11 = aS11 /aNbPix  - aS1 * aS1;
                     aS12 = aS12 /aNbPix  - aS1 * aS2;
                     aS22 = aS22 /aNbPix  - aS2 * aS2;

                     double aCor = aS12 / sqrt(ElMax(1e-5,aS11*aS22));
                     if (aCor>aCorMax)
                     {
                         aCorMax = aCor;
                         aDecMax = aDec;
                     }

                 }
             }
             //std::cout << "COR= " << aCorMax << "\n";
             anImDecX.SetVal(aX,aY,aDecMax.x);
             anImDecY.SetVal(aX,aY,aDecMax.y);
             // anImDecY.SetVal(aC1,aDecMax.y);
        }
    }


    anImDecX.Save("DecX.tif");
    anImDecY.Save("DecY.tif");


    return EXIT_SUCCESS;
}



// Visu polaire

int TD_Exo7(int argc,char ** argv)
{
    std::string aNameIm1;
    std::string aNameIm2;
    Pt2dr aDec0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image")
                    << EAMC(aDec0,"Name of image"),
        LArgMain()  
    );

    cTD_Im anImDecX = cTD_Im::FromString(aNameIm1);
    cTD_Im anImDecY = cTD_Im::FromString(aNameIm2);
   
    Pt2di aSzF = anImDecX.Sz();
    cTD_Im anImRho(aSzF.x,aSzF.y);
    cTD_Im anImTeta(aSzF.x,aSzF.y);

    int aBrd = 0;
    // Parcour des XY de l'image 1
    for (int aX = aBrd ; aX < aSzF.x -aBrd ; aX++)
    {
        for (int aY = aBrd ; aY < aSzF.y -aBrd ; aY++)
        {
             double aDX = anImDecX.GetVal(aX,aY) -aDec0.x;
             double aDY = anImDecY.GetVal(aX,aY) -aDec0.y;

             anImRho.SetVal(aX,aY,sqrt(aDX*aDX+aDY*aDY));
             
        }
    }

    anImRho.Save("Rho.tif");

    return EXIT_SUCCESS;
}





int TD_Exo8(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo9(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

#endif


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
