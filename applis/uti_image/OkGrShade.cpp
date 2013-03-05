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

Im2DGen AllocImGen(Pt2di aSz,const std::string & aName)
{
    return D2alloc_im2d(type_im(aName),aSz.x,aSz.y);
}





int main(int argc,char ** argv)
{
     std::string aNameIn;
     std::string aNameOut;
     Pt2di aP0(0,0),aSzGlob(0,0);
     INT aNbDir = 20;
     REAL aFZ = 1.0;

     REAL aPdsAnis = 0.95;
     INT  aBrd = -1;
     std::string aTMNt = "real4";
     std::string aTShade = "real4";
     INT aDequant =0;
     INT aVisu = 1;
     REAL aHypsoDyn = -1.0;
     REAL aHypsoSat = 0.5;
     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAM(aNameIn) ,
           LArgMain() << EAM(aNameOut,"Out",true)
                      << EAM(aVisu,"Visu",true)
                      << EAM(aP0,"P0",true)
                      << EAM(aSzGlob,"Sz",true)
                      << EAM(aFZ,"FZ",true)
                      << EAM(aPdsAnis,"Anisotropie",true)
		      << EAM(aNbDir,"NbDir",true)
		      << EAM(aBrd,"Brd",true)
                      << EAM(aTMNt,"TypeMnt",true)
                      << EAM(aTShade,"TypeShade",true)
                      << EAM(aDequant,"Dequant",true)
                      << EAM(aHypsoDyn,"HypsoDyn",true)
                      << EAM(aHypsoSat,"HypsoSat",true)
    );	

    if (aNameOut=="")
       aNameOut = StdPrefix(aNameIn) +std::string("Shade.tif");

     Tiff_Im aFileIn(aNameIn.c_str());
     if (aSzGlob== Pt2di(0,0))
        aSzGlob = aFileIn.sz();


     Im2DGen aMnt = AllocImGen(aSzGlob,aTMNt);
     REAL aVMin;
     if (aDequant)
     {
         ElImplemDequantifier aDeq(aSzGlob);
         aDeq.SetTraitSpecialCuv(true);
         aDeq.Dequant(aSzGlob, trans(aFileIn.in(),aP0),1);
         ELISE_COPY
         (
              aMnt.all_pts(),
              aDeq.ImDeqReelle() * aFZ,
              aMnt.out()
         );
/*
         Tiff_Im::CreateFromIm(aMnt,StdPrefix(aNameIn) +"Deq.tif");
         Tiff_Im::CreateFromIm(aDeq.DistMoins(),StdPrefix(aNameIn) +"DMoins.tif");
         Tiff_Im::CreateFromIm(aDeq.DistPlus(),StdPrefix(aNameIn) +"DPlus.tif");
     
*/
     }
     else 
         ELISE_COPY
         (
              aMnt.all_pts(),
              trans(aFileIn.in(),aP0)*aFZ,
              aMnt.out()|VMin(aVMin)
         );

     if (aBrd>0)
     {
        cout << "VMin = " << aVMin << "\n";
        ELISE_COPY(aMnt.border(aBrd),aVMin-1000,aMnt.out());
     }

     // Im2D_REAL4 aShade(aSzGlob.x,aSzGlob.y);
     Im2DGen aShade =  AllocImGen(aSzGlob,aTShade);
     ELISE_COPY(aShade.all_pts(),0,aShade.out());

     REAL aRatio = ElMin(800.0/aSzGlob.x,700.0/aSzGlob.y);
     Video_Win * pW  = aVisu                          ?
                       Video_Win::PtrWStd(aSzGlob*aRatio) :
                       0                              ;

     if (pW)
        pW = pW->PtrChc(Pt2dr(0,0),Pt2dr(aRatio,aRatio));

     REAL SPds = 0;
     REAL aSTot = 0;
     REAL Dyn = 1.0;
    if (aTShade != "u_int1")
       Dyn = 100;
     for (int aK=0 ; aK< 2 ; aK++)
     {
        SPds = 0;
        for (int i=0; i<aNbDir; i++)
        {
           REAL Teta  = (2*PI*i) / aNbDir ;
           Pt2dr U(cos(Teta),sin(Teta));
           Pt2di aDir = U * (aNbDir * 4);
           REAL Pds = (1-aPdsAnis) + aPdsAnis *ElSquare(1.0 - euclid(U,Pt2dr(0,1))/2);
           if (aK==1)
              Pds = (Pds*Dyn) / (2*aSTot);
           Symb_FNum Gr = (1-cos(PI/2-atan(gray_level_shading(aMnt.in()))))
                          *255.0;
           cout << "Dir " << i << " Sur " << aNbDir <<  " P= " << Pds << "\n";
           SPds  += Pds;
           if (aK==1)
           {
	      ELISE_COPY
	      (
	          line_map_rect(aDir,Pt2di(0,0),aSzGlob),
	          Min(255*Dyn,aShade.in()+Pds*Gr),
	            aShade.out() 
                  | (pW ? (pW->ogray()<<(aShade.in()/SPds)) : Output::onul())
              );
           }
        }
        aSTot  = SPds;
     }


     Fonc_Num aFoncRes = aShade.in()/SPds;
     if (aHypsoDyn >0)
         aFoncRes = its_to_rgb(Virgule(aFoncRes,trans(aFileIn.in(),aP0)*aHypsoDyn,255*aHypsoSat));
     // Tiff_Im::Create8BFromFonc(aNameOut,aShade.sz(),aShade.in()/SPds);
     Tiff_Im::Create8BFromFonc
     (
           aNameOut,
           aShade.sz(),
           aFoncRes
     );

     getchar();

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
