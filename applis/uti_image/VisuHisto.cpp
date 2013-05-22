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




class cHisto 
{
    public :
        cHisto(int argc,char ** argv);
        void NoOp() {}

    private :

        Pt2dr ToWinC(Pt2dr aPH)
        {
           return Pt2dr
                  (
                     (aPH.x-mDebH)*(double(mSzWX)/(mFinH-mDebH)),
                     mSzWY- aPH.y* (mSzWY)/double(mMaxH)
                  );
        }



        Pt2dr FromWinC(Pt2dr aPW)
        {
           return Pt2dr
                  (
                     mDebH + aPW.x * ((mFinH-mDebH)/double(mSzWX)),
                     (mSzWY-aPW.y) * (double(mMaxH)/mSzWY)
                  );
        }

        int W2H(double anX)
        {
           Pt2dr aPH =  FromWinC(Pt2dr(anX,0));
           return  ElMax(0,ElMin(mVMax-1,round_ni(aPH.x)));
        }

        int  mSzWX;
        int  mSzWY;
        
        Im1D_INT4   mHisto;
        Im1D_REAL8  mHCum;
        int         mMaxH;
        int         mDebH;
        int         mFinH;
        int         mVMax;

        Video_Win * mW;
};

cHisto::cHisto(int argc,char ** argv) :
    mHisto (1),
    mHCum  (1)
{

    std::string aNameIm;
    mVMax = 1<<16;
    mSzWX = 500;
    mSzWY = 300;
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIm),
	LArgMain()  << EAM(mVMax,"VMax",true)
    );	
    mW = Video_Win::PtrWStd(Pt2di(mSzWX,mSzWY+10));
    Tiff_Im aFileIn = Tiff_Im::StdConv(aNameIm.c_str());

    mHisto = Im1D_INT4(mVMax,0);
    mHCum  = Im1D_REAL8(mVMax,0.0);
    ELISE_COPY
    (
        aFileIn.all_pts().chc(Max(0,Min(mVMax-1,aFileIn.in()))),
        1,
        mHisto.histo()
    );
    mHCum.data()[0] = mHisto.data()[0];
    for (int aK=1 ; aK<mVMax ; aK++)
       mHCum.data()[aK] = mHCum.data()[aK-1]+mHisto.data()[aK];
    for (int aK=0 ; aK<mVMax ; aK++)
       mHCum.data()[aK] /= mHCum.data()[mVMax-1];
    

    ELISE_COPY
    (
         select(mHisto.all_pts(),mHisto.in()>0),
         mHisto.in(),
            VMax(mMaxH)
         |  (VMin(mDebH)|VMax(mFinH))<<FX
         
    );
    mFinH++;
    
    for (int anX=0 ; anX <mSzWX ; anX++)
    {
       // Pt2dr aPH =  FromWinC(Pt2dr(anX,0));
       // int aHX = ElMax(0,ElMin(mVMax-1,round_ni(aPH.x)));
       int aHX = W2H(anX);
       int aHY = mHisto.data()[aHX];

        Pt2dr aP0 = ToWinC(Pt2dr(aHX,0));
        Pt2dr aP1 = ToWinC(Pt2dr(aHX,aHY));
        mW->draw_seg(aP0,aP1,mW->pdisc()(P8COL::red));
    }

    std::cout  << "Debut " << mDebH << " Fin " << mFinH << "\n";
    while(1)
    {
        Clik  aCl = mW->clik_in();
        int anX = round_ni(aCl._pt.x);
        anX = ElMax(0,ElMin(anX,mSzWX));
        Pt2dr aP0 = Pt2dr(anX,0);
        Pt2dr aP1 = Pt2dr(anX,mSzWY);
        mW->draw_seg(aP0,aP1,mW->pdisc()(P8COL::blue));
        int aRad =  W2H(anX);
        int aPop =  mHisto.data()[ aRad];
        double aCum =  mHCum.data()[ aRad];
        std::cout << " X " << aCl._pt.x << "\n";
        std::cout << "Rad = " << aRad 
                  << " Cumul = " << ( aCum *100) << "% " 
                  << " Pop = " << aPop  << "\n";
    }
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
