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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/


#include "StdAfx.h"





class cCorCamL : public cAppliWithSetImage
{
   public :

       cCorCamL(int argc,char** argv);

       void DoOne(const std::string & aName);

       std::string mFullName;
       double mGama;
       double mDif;
       bool   mVisu;
};

cCorCamL::cCorCamL(int argc,char** argv) :
   cAppliWithSetImage(argc-1,argv+1,TheFlagNoOri),
   mGama             (0.55),
   mDif              (0),
   mVisu             (false)
{
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile),
        LArgMain()  << EAM(mGama,"Gama",true,"Gama correc to invert, Def=0.55")
                    << EAM(mDif,"Dif",true,"Out Orientation, if unspecified : calc")
                    << EAM(mVisu,"Visu",true,"Visualisation, Def=false")
   );

   const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();

   if (aSetIm->size()==1)
   {
      DoOne((*aSetIm)[0]);
   }
   else if (aSetIm->size() >1)
   {
        ExpandCommand(3,"",true);
   }
}



void cCorCamL::DoOne(const std::string & aName)
{

    Im2D_REAL4 anIm0 =  Im2D_REAL4::FromFileStd(aName);
    Pt2di aSz = anIm0.sz();
    Im2D_REAL4 anImEg(aSz.x,aSz.y);
    ELISE_COPY(anIm0.all_pts(),pow(anIm0.in(),1/mGama),anImEg.out());

    double aMaxDif= 100;
    double aStep =  1;
    int aNbNiv = aMaxDif / aStep;

    Im1D_INT4   aH(1+2*aNbNiv,0);
    Fonc_Num aFDif = anImEg.in() -trans(anImEg.in_proj(),Pt2di(0,-1));
    aFDif = round_ni(aFDif /aStep);
    aFDif = Max(0,Min(2*aNbNiv,aFDif +aNbNiv));
    ELISE_COPY(select(anImEg.all_pts(),FY%2).chc(aFDif),1,aH.histo());

    if (! EAMIsInit(&mDif))
    {
       std::vector<int> aVH;
       for (int aK=0 ; aK<=2*aNbNiv ; aK++)
          aVH.push_back(aH.data()[aK]);
       mDif = GetValPercOfHisto(aVH,50) - aNbNiv;

       std::cout << aName <<  " : DifMed  = " << mDif << "\n";
    }


    if (mVisu)
    {
       {
           Pt2dr aSzMax(1200,800);
           double aRatio  = ElMin(aSz.x/aSzMax.x, aSz.y/aSzMax.y);
           Pt2di aSzW = Pt2di(Pt2dr(aSz)/aRatio);
           Video_Win aWIm = Video_Win::WStd(aSzW,1.0);
           Im2D_U_INT1 aImD0(aSz.x,aSz.y);
           // ELISE_COPY(aImD0.all_pts(),aImD0.out());

           // ELISE_COPY(aWIm.all_pts(),StdFoncChScale(anIm0.in_proj(),Pt2dr(0,0),Pt2dr(aRatio,aRatio)),aWIm.ogray());
           
           ELISE_COPY
           (
                aWIm.all_pts(),
                StdFoncChScale(anIm0.in_proj(),Pt2dr(0,0),Pt2dr(aRatio,aRatio)),
                aWIm.ogray()
           );
           
       }


       std::cout << " H0 " << aH.data()[aNbNiv] << "\n";
       Pt2di SZ = Pt2di(1+2*aNbNiv,200) * 3;
       Video_Win aW = Video_Win::WStd(SZ,1.0);
       Disc_Pal  Pdisc =aW.pdisc();

       Plot_1d  Plot1
                (
                        aW,
                        Line_St(Pdisc(P8COL::green),3),
                        Line_St(Pdisc(P8COL::black),2),
                        Interval(-aNbNiv,aNbNiv),
                           NewlArgPl1d(PlBox(Pt2dr(3,3),Pt2dr(SZ)-Pt2dr(3,3)))
                        + Arg_Opt_Plot1d(PlScaleY(1.0))
                        + Arg_Opt_Plot1d(PlBoxSty(Pdisc(P8COL::blue),3))
                        + Arg_Opt_Plot1d(PlClipY(true))
                        + Arg_Opt_Plot1d(PlModePl(Plots::draw_fill_box))
                        + Arg_Opt_Plot1d(PlClearSty(Pdisc(P8COL::white)))
                        + Arg_Opt_Plot1d(PlotFilSty(Pdisc(P8COL::red)))
                   );
       Plot1.clear();

       int aVMax;
       ELISE_COPY(aH.interior(1),aH.in(),VMax(aVMax));
       // ELISE_COPY(Plot1.all_pts(),FX,Plot1.out());
       ELISE_COPY(Plot1.all_pts(),trans(aH.in(0), aNbNiv)*(100.0/aVMax),Plot1.out());
       Plot1.show_axes();
       Plot1.show_box();
       aW.clik_in();

    }

    int aParite=0;
    if (mDif>0)
    {
       ELISE_COPY(select(anImEg.all_pts(),(FY%2)==aParite),anImEg.in()+mDif,anImEg.out());
    }
    else
    {
       ELISE_COPY(select(anImEg.all_pts(),(FY%2)==(1-aParite)),anImEg.in()-mDif,anImEg.out());
    }
    ELISE_COPY(anImEg.all_pts(),pow(anImEg.in(), mGama),anIm0.out());

    Tiff_Im::Create8BFromFonc("Eq_"+aName,aSz,anIm0.in());
}
 

int CCL_main (int argc,char** argv)
{
    cCorCamL(argc,argv);
    return 1;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
