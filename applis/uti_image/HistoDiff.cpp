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



#define DEF_OFSET -12349876

class cHisto 
{
    public :
        cHisto(int argc,char ** argv);
        void NoOp() {}

        int ToYW(double aV)
        {
            return mTWY+mBrdY- round_ni(mBrdY/2+aV*mFactY);
        }
        int HAbsToYW(double aV)
        {
            return mTWY+mBrdY- round_ni(mBrdY/2+aV*mFactYHAbs);
        }
    private :
        
        Im1D_REAL8  mHAbs;
        double mFactYHAbs ;
        double mPop ;
        double mStep ;
        double mFactY ;
        int    mBrdY;
        int    mTWY ;
};

cHisto::cHisto(int argc,char ** argv) :
    mHAbs (1)
{

    std::string aNameIm1;
    std::string aNameIm2;

    mStep = 0.1;
    double aVMax = 40;
    mTWY = 200;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIm1)
                    << EAM(aNameIm2),
	LArgMain()  << EAM(mStep,"Step",true)
    );	

    std::string aDir1,aNF1;
    SplitDirAndFile(aDir1,aNF1,aNameIm1);
    std::string aDir2,aNF2;
    SplitDirAndFile(aDir2,aNF2,aNameIm2);

    std::string aNameFileDif =  aDir1 + "Dif_" + StdPrefix(aNF1)
                               + "___" + StdPrefix(aNF2) + ".tif";


    Tiff_Im tiff1 = Tiff_Im::StdConv(aNameIm1.c_str());
    Tiff_Im tiff2 = Tiff_Im::StdConv(aNameIm2.c_str());

    Symb_FNum aDifI (round_ni((tiff1.in()-tiff2.in()+aVMax)/mStep));
    Symb_FNum aDifIA (round_ni(Abs(tiff1.in()-tiff2.in())/mStep));

    int aMaxIAbs  = round_ni(aVMax/mStep);
    int aMaxI  = 2*aMaxIAbs;
    Im1D_REAL8  aHisto(aMaxI,0.0);
    mHAbs = Im1D_REAL8(aMaxIAbs,0.0);

    ELISE_COPY
    (
       tiff1.all_pts(),
       1,
          aHisto.histo(true).chc(aDifI)
       |  mHAbs.histo(true).chc(aDifIA)
    );
    int aSzHAbs = aMaxIAbs-1;
    for (int aK=1; aK <=  aSzHAbs; aK++)
    {
        mHAbs.data()[aK] +=  mHAbs.data()[aK-1];
   // std::cout <<  mHAbs.data()[aK]  << " " << mHAbs.data()[aK-1] << "\n";
    }
    mPop =  mHAbs.data()[aSzHAbs];

    double aHMax = 0.0;
    ELISE_COPY(aHisto.all_pts(),aHisto.in(),VMax(aHMax));

    
    mBrdY = 10;
    Video_Win aW = Video_Win::WStd(Pt2di(aMaxI,mTWY+mBrdY),1.0);

    aW.draw_seg
    (
          Pt2di(0,ToYW(0)),
          Pt2di(aMaxI,ToYW(0)),
          aW.pdisc()(P8COL::green)
    );
    for (int aV=-round_ni(aVMax) ; aV < round_ni(aVMax); aV++)
    {
       int anX = round_ni((aV+aVMax)/mStep);
       aW.draw_seg
       (
             Pt2di(anX,mTWY),
             Pt2di(anX, mTWY+mBrdY),
             aW.pdisc()(P8COL::blue)
       );
    }
    aW.draw_seg
    (
          Pt2di(aMaxI/2,0),
          Pt2di(aMaxI/2,mTWY+mBrdY),
          aW.pdisc()(P8COL::green)
    );

    mFactY = mTWY/aHMax;
    mFactYHAbs = mTWY/ mPop;
    double aSZ=0,aSZ2=0,aSAbsZ=0,aSP=0;
    for (int aV=0 ; aV <aMaxI-2 ; aV++)
    {

        // Pt2di aP0(aV, mTWY+10- round_ni(5+aHisto.data()[aV]*mFactY));
        // Pt2di aP1(aV+1, mTWY+10-round_ni(5+aHisto.data()[aV+1]*mFactY));

        Pt2di aP0(aV, ToYW(aHisto.data()[aV]));
        Pt2di aP1(aV+1, ToYW(aHisto.data()[aV+1]));
        // Pt2di aP1(aV+1, mTWY+10-round_ni(5+aHisto.data()[aV+1]*mFactY));
        aW.draw_seg(aP0,aP1,aW.pdisc()(P8COL::red));
        double aP = aHisto.data()[aV];
        double aZ = aV*mStep - aVMax;
        aSP += aP;
        aSZ += aZ * aP;
        aSZ2 += ElSquare(aZ) * aP;
        aSAbsZ += ElAbs(aZ) * aP;
        if (aV>=aMaxIAbs)
        {
           int aVH = aV-aMaxIAbs;
           Pt2di aQ0(aV   , HAbsToYW(mHAbs.data()[aVH]));
           Pt2di aQ1(aV+1 , HAbsToYW(mHAbs.data()[aVH+1]));
           aW.draw_seg(aQ0,aQ1,aW.pdisc()(P8COL::white));
        }
    }
    for (int aK=0 ; aK<50 ; aK++)
    {
       std::cout << "Dz = " << (aK*mStep)
                 << " Err = " << (100.0 * (mHAbs.data()[aK]/mPop))
                 << "\n";
    }
    std::cout <<  " ; Moy= " << aSZ/aSP 
              <<  " ; Ec2= " << sqrt(aSZ2/aSP)
              <<  " ; EcAbs= " << aSAbsZ/aSP
              << " \n";

    Tiff_Im  aFileDif
             (
                 aNameFileDif.c_str(),
                 tiff1.sz(),
                 GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    ELISE_COPY
    (
        aFileDif.all_pts(),
        128+(tiff1.in()- tiff2.in())/mStep,
/*
        its_to_rgb
        (
             Virgule
             (
                 128+(tiff1.in()- tiff2.in())/mStep,
                 tiff1.in(),
                 0.4
             )
        ),
*/
        aFileDif.out()
    );
    getchar();
}


int main(int argc,char ** argv)
{
  cHisto aH(argc,argv);
  aH.NoOp();
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
