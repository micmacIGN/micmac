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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>


static Pt2di aSzTileMasq(1000000,1000000);


#define DEF_OFSET -12349876

typedef enum
{
   eModCst,
   eModTimbre,
   eModRand,
   eModRandT5,
   eModGeom,
   eModCroix
} eModePat;

class cPat
{
    public :
        cPat(Pt2di aSz,Fonc_Num aF) :
            mPat (aSz.x,aSz.y)
        {
            int aSF,aS1;
            ELISE_COPY
            (
               mPat.all_pts(),
               Virgule(aF,1),
               Virgule(mPat.out() | sigma(aSF),sigma(aS1))
            );
            mDens = aSF / double(aS1);
        }
        void SetDens(double aDens)
        {
            if (mDens<=0.5)
            {
                mV1 = (aDens<0.5) ? 255 : 0;
                mV0 = ((255*aDens) - mV1*mDens)/(1-mDens);
            }
            else
            {
                mV0 = (aDens<0.5) ? 255 : 0;
                mV1 = ((255*aDens) - mV0*(1-mDens))/mDens;
            }
            mV0 = ElMin(255.0,ElMax(0.0,mV0));
            mV1 = ElMin(255.0,ElMax(0.0,mV1));
        }
        int GetV(Pt2di aP)
        {
           return round_ni(mPat.get(aP.x,aP.y) ? mV1 : mV0);
 
        }
    private :
       Im2D_Bits<1> mPat;
       double mDens;
       double mV0;
       double mV1;
};


class cAppliTramage
{
    public :
       cAppliTramage(int argc,char ** argv,int aCanal,cAppliTramage * = 0);

       void NoOp() {}
       void MakeSingleImage();
       void MakeSingleImage(cAppliTramage &,cAppliTramage &);

       int mNbCh;
    private :
       void MakePatch(Pt2dr aPOMur,Pt2dr aSzMur,const std::string & aNameAdd);
 
       Tiff_Im FileRes (Pt2di aSz,const std::string &  aName,bool IsCol);

       std::string NameOut(const std::string & anAdd);


       Im2D_U_INT1   mIPat;
       Im2D_U_INT1   mIRes;
       double        mResolTif;
       std::string   mNameIn;
       std::string   mNameOut;
       Pt2di  mSzBin;
};


void cAppliTramage::MakePatch(Pt2dr aPOMur,Pt2dr aSzMur,const std::string & aNameAdd)
{
   Pt2di aPOIm = aPOMur / mResolTif;
   Pt2di aSzIm = aSzMur / mResolTif;

   Tiff_Im aFRes = FileRes(aSzIm,NameOut(aNameAdd),false);
   ELISE_COPY
   (
        aFRes.all_pts(),
        trans(mIRes.in(),aPOMur),
        aFRes.out()
   );
}

Tiff_Im cAppliTramage::FileRes (Pt2di aSz,const std::string &  aName,bool isColored)
{
    return Tiff_Im 
            (
                  aName.c_str(),
                  aSz,
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  (isColored ? Tiff_Im::RGB : Tiff_Im::BlackIsZero),
		     Tiff_Im::Empty_ARG
                  +  Arg_Tiff(Tiff_Im::ANoStrip())
                  +  Arg_Tiff(Tiff_Im::AFileTiling(aSzTileMasq))
                  +  Arg_Tiff(Tiff_Im::AResol(mResolTif,Tiff_Im::Cm_Unit))
            );
}


std::string cAppliTramage::NameOut(const std::string & anAdd)
{
    std::string aDir,aFile;
    SplitDirAndFile(aDir,aFile,mNameIn);


    return  aDir + "Tramed" + anAdd + ".tif";
}

cAppliTramage::cAppliTramage(int argc,char ** argv,int aCanal,cAppliTramage * aMaster) :
   mIPat (1,1),
   mIRes (1,1)
{

   Pt2dr aSzMur(350.0,295.0); // Cm
   double aPasPattern = 1.5;  // Cm

   int  aSzGrain = 21;
   double aGama = 1.0;

   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAM(mNameIn),
       LArgMain()  << EAM(mNameOut,"Out",true)
                   << EAM(aGama,"Gama",true)
                   << EAM(aPasPattern,"ResolPattern",true)
   );

    mResolTif = aPasPattern /aSzGrain;

    Pt2di aSzMax = round_ni(aSzMur /aPasPattern);

    //    Parametres 

    Tiff_Im aTif(mNameIn.c_str());
    mNbCh = aTif.nb_chan();

    Pt2di aSzFileIn = aTif.sz();
    double  aResol = ElMin
                     (
                        aSzFileIn.x/double(aSzMax.x),
                        aSzFileIn.y/double(aSzMax.y)
                     );
    Pt2di aSzIn
          (
	       round_ni(aResol*aSzMax.x),
	       round_ni(aResol*aSzMax.y)
          );
    Pt2di aSzOutGr = round_ni(aSzIn/aResol);


    Pt2di aP0  =  (aSzFileIn-aSzIn)/2;
    Pt2di aP1  = aP0 + aSzIn;


    Im2D_REAL8 aTimbre(aSzGrain,aSzGrain);

    if (mNameOut == "")
    {
        mNameOut = NameOut("");
/*
std::string cAppliTramage::NameOut(const std::string & anAdd)
	std::string aDir,aFile;
        SplitDirAndFile(aDir,aFile,mNameIn);

        mNameOut = aDir + "Tramed.tif";
*/
    }

    //  Calcul de l'image en niveau de gris


    std::cout << aSzOutGr << aResol << "\n";

    Video_Win *  aW = Video_Win::PtrWStd(aSzOutGr);
    Im2D_U_INT1 aIGr(aSzOutGr.x,aSzOutGr.y);

    Fonc_Num Fori =  StdFoncChScale
                     (
                         aTif.in_proj().kth_proj(aCanal),
                         Pt2dr(aP0.x,aP0.y),
                         Pt2dr(aResol,aResol)
                     );

    Fori = Max(0,Min(255,255 * pow(Fori/255.0,1/aGama)));
    // Fori = 128;
    ELISE_COPY
    (
        aIGr.all_pts(),
        Fori,
        aIGr.out() | aW->ogray()
    );

    ELISE_COPY
    (
        aTimbre.all_pts(),
        StdFoncChScale
        (
             aIGr.in_proj(),
             Pt2dr(0,0),
             Pt2dr(aSzOutGr)/double(aSzGrain)
        ),
        aTimbre.out() 
    );

    double aST;
    {
       double aS1;
       ELISE_COPY
       ( 
          aTimbre.all_pts(),
          Virgule(aTimbre.in(),1.0),
          Virgule(sigma(aST),sigma(aS1))
       );
       aST /= aS1;
    }

    Im2D_REAL8 aTimbre5(aSzGrain*5,aSzGrain*5);
    ELISE_COPY
    (
        aTimbre5.all_pts(),
        StdFoncChScale
        (
             aIGr.in_proj(),
             Pt2dr(0,0),
             Pt2dr(aSzOutGr)/double(aSzGrain*5)
        ),
        aTimbre5.out() 
    );


    double aC = (aSzGrain-1)/2.0;
    double aCI = round_ni(aC);
    Im2D_Bits<1> aICr(aSzGrain,aSzGrain);
    ELISE_COPY
    (
          aICr.all_pts(),
              (Abs(FX-aCI)<1.7)
          ||  (Abs(FY-aCI)<1.7),
          aICr.out()
    );

    cPat aPatCr
         (
             Pt2di(aSzGrain,aSzGrain),
             (Abs(FX-aC)<1.7)||(Abs(FY-aC)<1.7)
         ) ;
    std::vector <cPat *>  aVPatNum;
    for (char aC= '0' ; aC<='9' ; aC++)
    {
       Im2D_Bits<1> aImC= cElBitmFont::BasicFont_10x8().ImChar(aC);
       aVPatNum.push_back
       (
          new cPat 
          (
             Pt2di(aSzGrain,aSzGrain),
             aImC.in(0)[Virgule((FX-1)/2,(FY-1)/2-1)]
          ) 
       );
    }


    //  Calcul de l'image binaire
    mSzBin = aSzOutGr * aSzGrain;
    mIRes = Im2D_U_INT1(mSzBin.x,mSzBin.y);

    mIPat = Im2D_U_INT1(aSzOutGr.x,aSzOutGr.y);
    for (int  anYGr=0 ; anYGr<aSzOutGr.y ; anYGr++)
    {
        ELISE_COPY
        (
           rectangle(Pt2di(0,anYGr),Pt2di(aSzOutGr.x,anYGr+1)),
           255-aIGr.in(),
           aW->ogray()
        );
        for (int  anXGr=0 ; anXGr<aSzOutGr.x ; anXGr++)
        {
             int aGr = aIGr.data()[anYGr][anXGr];
             double aDens =  aGr / 255.0;

             bool aPMarq=  ((anXGr%9)<3) && ((anYGr%9)<3);
             int xind = (anXGr%3);
             int yind = (anYGr%3);

             int xindG = (anXGr/3)%2;
             int yindG = (anYGr/3)%2;

             bool   aPattSq = NRrandom3() > 0.5 ;//(xindG+yindG)==1;
             bool   aPattInv  = (aDens < 0.5);

             //if ((anXGr/3) %2) aPattSq = ! aPattSq;
             //if ((anYGr/3) %2) aPattInv = ! aPattInv;


             double aR2Max =   ElSquare(aC) 
                             * (aPattInv ? aDens : (1-aDens))
                             * (aPattSq  ? 1 : (4.0/PI)) ;

             eModePat aMod = eModGeom;
             cPat * aCurPat=0;
             if ((xind==1)||(yind==1))
             {
                if ((xind==1)&&(yind==1))
                   aMod = eModTimbre;
                else  if ((xind+yind==1))
                {
                   if (xind==0)
                       aMod = eModCst;
                   else
                       aMod = eModRand;
                }
                else
                   aMod = eModRandT5;
                if (aPMarq)
                {
                   aMod = eModCroix;

                   if (xind != 1)
                      aCurPat = aVPatNum[(anYGr/9) %10];
                   else if (yind != 1)
                      aCurPat = aVPatNum[(anXGr/9) %10];
                   else
                      aCurPat = & aPatCr;

                   aCurPat->SetDens(aDens);
                }
             }
             double aSTR5=0;
             int x0= round_ni(NRrandom3()*(4*aSzGrain-1));
             int y0= round_ni(NRrandom3()*(4*aSzGrain-1));
             if (aMod==eModRandT5)
             {
                  double aS1;
                  Pt2di aP0 (x0,y0);
                  ELISE_COPY
                  ( 
                      rectangle(aP0,aP0+Pt2di(aSzGrain,aSzGrain)),
                      Virgule(aTimbre5.in(),1.0),
                      Virgule(sigma(aSTR5),sigma(aS1))
                  );
                  aSTR5 /= aS1;
             }

             if (aMaster) 
                aMod = (eModePat) aMaster->mIPat.data()[anYGr][anXGr];
             for (int aDx=0 ; aDx<aSzGrain ; aDx++)
                 for (int aDy=0 ; aDy<aSzGrain ; aDy++)
                 {

                     int aXres = anXGr * aSzGrain + aDx;
                     int aYres = anYGr * aSzGrain + aDy;


                     switch (aMod)
                     {
                        case eModTimbre :
                        {
                           double aC = aTimbre.data()[aDy][aDx] * (255/aST) * aDens;
                           mIRes.data()[aYres][aXres] = ElMin(255,round_ni(aC));
                        }
                        break;

                        case eModCst :
                           mIRes.data()[aYres][aXres] = (U_INT1)(255 * aDens);
                        break;

                        case eModRand :
                           mIRes.data()[aYres][aXres] = 255 * (NRrandom3() <aDens);
                        break;

                        case eModGeom :
                        {
                           Pt2dr  aDif (aDx-aC,aDy-aC);
                           double aR2 =   (aPattSq) ? ElSquare(dist8(aDif)) : square_euclid(aDif);
                           bool aCoul = aPattInv ? (aR2 <= aR2Max) :(aR2 > aR2Max);

			   if ((dist4(aDif) <=1.1) && (aR2Max > 9))
                              aCoul = ! aCoul;

				   
                           mIRes.data()[aYres][aXres] = aCoul*255;
                       }
                       break;

                       case eModRandT5 :
                       {
                           double aC = aTimbre5.data()[y0+aDy][x0+aDx] * (255/aSTR5) * aDens;
                           mIRes.data()[aYres][aXres] = ElMin(255,round_ni(aC));
                       }
                       break;


                       case eModCroix :
                       {
                           // mIRes.data()[aYres][aXres] = 255 * aICr.get(aDx,aDy);
                           mIRes.data()[aYres][aXres] = aCurPat->GetV(Pt2di(aDx,aDy));
                       };
                 }
            }
            mIPat.data()[anYGr][anXGr] = aMod;
        }
    }

/*
    Tiff_Im aFileRes  = FileRes(mSzBin,mNameOut);
    ELISE_COPY(mIRes.all_pts(),mIRes.in(),aFileRes.out());

    MakePatch(Pt2dr(10,10),Pt2dr(15,20),"_P1");
    MakePatch(Pt2dr(10,10),Pt2dr(10,15),"_P0");
*/
}

void cAppliTramage::MakeSingleImage()
{
    Tiff_Im aFileRes  = FileRes(mSzBin,mNameOut,false);
    ELISE_COPY(mIRes.all_pts(),mIRes.in(),aFileRes.out());

    MakePatch(Pt2dr(10,10),Pt2dr(15,20),"_P1");
    MakePatch(Pt2dr(10,10),Pt2dr(10,15),"_P0");
}

void cAppliTramage::MakeSingleImage(cAppliTramage & anAp1,cAppliTramage & anAp2)
{
    Tiff_Im aFileRes  = FileRes(mSzBin,mNameOut,true);
    ELISE_COPY
    (
        mIRes.all_pts(),
        Virgule(mIRes.in(),anAp1.mIRes.in(),anAp2.mIRes.in()),
        aFileRes.out()
    );
}




int main(int argc,char ** argv)
{
    cAppliTramage anAppli(argc,argv,0);
    if (anAppli.mNbCh==3)
    {
        cAppliTramage anAp1(argc,argv,1,&anAppli);
        cAppliTramage anAp2(argc,argv,2,&anAppli);
         anAppli.MakeSingleImage(anAp1,anAp2);
    }
    else
    {
       anAppli.MakeSingleImage();
    }
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
