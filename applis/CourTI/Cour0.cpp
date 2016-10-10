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






void FiltreSimple()
{

    Tiff_Im aImEg("/home/mpierrot/Data/COUR/E10.tif");
    Pt2di aSz = aImEg.sz();

    Im2D_U_INT1 aImMoy3(aSz.x,aSz.y);


    ELISE_COPY
    (
       aImEg.all_pts(),
       rect_som(aImEg.in_proj(),1)/9.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/Moy3x3.tif");


    ELISE_COPY
    (
       aImEg.all_pts(),
       rect_som(aImEg.in_proj(),3)/49.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/Moy7x7.tif");


    ELISE_COPY
    (
       aImEg.all_pts(),
       rect_som(aImEg.in_proj(),Pt2di(1,3))/21.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/Moy3x7.tif");



    Im2D_REAL8 aIGX(3,3,
                       " 0 0 0"
                       " -1 0 1"
                       " 0 0 0"
                  );
    ELISE_COPY
    (
         aImEg.all_pts(),
         Max(0,Min(255,128+4*som_masq(aImEg.in_proj(),aIGX))),
         aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/GradSimple.tif");





    Im2D_REAL8 aISobX(3,3,
                       " 1 0 -1"
                       " 2 0 -2"
                       " 1 0 -1"
                  );
    ELISE_COPY
    (
         aImEg.all_pts(),
         Max(0,Min(255,128+som_masq(aImEg.in_proj(),aISobX))),
         aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/SobelX.tif");


    Im2D_REAL8 aISobY(3,3,
                       " -1 -2  -1"
                       " 0 0  0"
                       " 1 2  1"
                  );
    ELISE_COPY
    (
         aImEg.all_pts(),
         Max(0,Min(255,128+som_masq(aImEg.in_proj(),aISobY))),
         aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/SobelY.tif");


    Im2D_REAL8 aLapl(3,3,
                       " 0  -1 0  "
                       " -1  4  -1"
                       "  0 -1 0 "
                  );
    ELISE_COPY
    (
         aImEg.all_pts(),
         Max(0,Min(255,aImEg.in()+ som_masq(aImEg.in_proj(),aLapl))),
         aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/LaplContraste.tif");
}

void EffetDeBord()
{

    Tiff_Im aImEg("/home/mpierrot/Data/COUR/Detail.tif");
    Pt2di aSz = aImEg.sz();

    Im2D_U_INT1 aImMed(aSz.x,aSz.y,0);
    ELISE_COPY(aImMed.all_pts(),255,aImMed.out());
    for (int aK=0 ; aK< 20 ; aK++)
    {
        ELISE_COPY(aImMed.border(3),0,aImMed.out());

        ELISE_COPY
        (
           aImMed.interior(3),
           // rect_median(aImMed.in(),3,256),
       rect_som(aImMed.in(),3)/49.0,
           aImMed.out()  
        );

        if ((aK%5)==4)
           Tiff_Im::CreateFromIm(aImMed,"/home/mpierrot/Data/COUR/EffBordMed0_"+ToString(aK)+".tif");
    }
    

    Im2D_U_INT1 aImMoy3(aSz.x,aSz.y,0);

    ELISE_COPY
    (
       aImEg.interior(3),
       rect_som(aImEg.in(),3)/49.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/EffBordTronquetif");


    ELISE_COPY
    (
       aImEg.all_pts(),
       rect_som(aImEg.in(0),3)/49.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/EffBord0.tif");


    Im2D_U_INT1 aImCoup(aSz.x+10,aSz.y+10,0);
    ELISE_COPY
    (
       aImCoup.all_pts(),
       trans(aImEg.in_proj(),Pt2di(-5,-5)),
       aImCoup.out()  
    );
    Tiff_Im::CreateFromIm(aImCoup,"/home/mpierrot/Data/COUR/ImInProj.tif");



    ELISE_COPY
    (
       aImEg.all_pts(),
       rect_som(aImEg.in_proj(),3)/49.0,
        aImMoy3.out()  
    );
    Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/EffBordInProj.tif");

}


void Contraste()
{
    Tiff_Im aImEg("/media/MYPASSPORT/Archi/Villesavin/CR2/img_2705_MpDcraw16B_GB_Scaled.tif");
    Pt2di aSz = aImEg.sz();

    int aNbV[5] = {16,16,16,16,16};
    for (int aK=0 ; aK<5 ; aK++)
    {
         Im2D_U_INT1 aImMoy3(aSz.x,aSz.y,0);
         Fonc_Num aF1 = Rconv(aImEg.in_proj());
         Fonc_Num aF2 = Square(Rconv(aImEg.in_proj()));
         for (int aT=0 ; aT<=aK ; aT++)
         {
             aF1 =  rect_som(aF1,aNbV[aK])/ElSquare(1.0+2*aNbV[aK]);
             aF2 =  rect_som(aF2,aNbV[aK])/ElSquare(1.0+2*aNbV[aK]);
         }

         aF2 = aF2 - Square(aF1);
         Fonc_Num aF = 255 *erfcc((aImEg.in_proj()-aF1) / (1.5*sqrt(Max(0.01,aF2))));


         ELISE_COPY(aImMoy3.all_pts(),Max(0,Min(255,aF)),aImMoy3.out());
         Tiff_Im::CreateFromIm(aImMoy3,"/media/MYPASSPORT/Archi/Villesavin/CR2/CJ_Contraste_" + ToString(aK) +".tif");
    }

}




void ItereMoy()
{
    Pt2di aSz(200,200);

    Im2D_U_INT1 aImIn(aSz.x,aSz.y,0);
    ELISE_COPY(rectangle(Pt2di(50,50),Pt2di(150,150)),255,aImIn.out());


    Tiff_Im::CreateFromIm(aImIn,"/home/mpierrot/Data/COUR/Rect.tif");

    int aNbV[4] = {16,11,9,8};
    for (int aK=0 ; aK<4 ; aK++)
    {
         Im2D_U_INT1 aImMoy3(aSz.x,aSz.y,0);
         Fonc_Num aF = aImIn.in_proj();
         for (int aT=0 ; aT<=aK ; aT++)
             aF =  rect_som(aF,aNbV[aK])/ElSquare(1.0+2*aNbV[aK]);

         ELISE_COPY(aImMoy3.all_pts(),Max(0,Min(255,aF)),aImMoy3.out());
         Tiff_Im::CreateFromIm(aImMoy3,"/home/mpierrot/Data/COUR/RectMoy_" + ToString(aK) +".tif");
    }
}


void CannyDeriche()
{
    Pt2di aSzR(200,200);
    Im2D_U_INT1 aImRect(aSzR.x,aSzR.y,0);
    ELISE_COPY(rectangle(Pt2di(50,50),Pt2di(150,150)),255,aImRect.out());


    double aExp[4] = {0.9,0.85,0.83,0.8};
    for (int aK=0 ; aK<4 ; aK++)
    {
         Im2D_U_INT1 aImMoyExp(aSzR.x,aSzR.y,0);

         Fonc_Num aF = aImRect.in_proj();
         for (int aT=0 ; aT<=aK ; aT++)
             aF =  canny_exp_filt(aF,aExp[aK],aExp[aK]) /canny_exp_filt(aImRect.inside(),aExp[aK],aExp[aK]);

         ELISE_COPY(aImMoyExp.all_pts(),Max(0,Min(255,aF)),aImMoyExp.out());
         Tiff_Im::CreateFromIm(aImMoyExp,"/home/mpierrot/Data/COUR/ExpRectMoy_" + ToString(aK) +".tif");
    }


    Tiff_Im aImEg("/home/mpierrot/Data/COUR/E10.tif");
    Pt2di aSz = aImEg.sz();

    Im2D_U_INT1 aImGX(aSz.x,aSz.y,0);
    Im2D_U_INT1 aImGY(aSz.x,aSz.y,0);
    Im2D_U_INT1 aImGXY(aSz.x,aSz.y,0);

    double alpha[3] = {0.5,1.0,2.0};
    for (int aK=0 ; aK<3 ; aK++)
    {
          {
          Fonc_Num aDer = deriche(aImEg.in_proj(),alpha[aK],10);
          ELISE_COPY
          (
               aImEg.all_pts(),
               Max(0,Min(255,128+5*aDer)),
               Virgule(aImGX.out(),aImGY.out())
          );
          }
          Fonc_Num aDerX = deriche(aImEg.in_proj(),alpha[aK],10);
          Fonc_Num aDerY = deriche(aImEg.in_proj(),alpha[aK],10);
          ELISE_COPY
          (
               aImEg.all_pts(),
               Max(0,Min(255,15 *sqrt(Square(aDerX.v0())+Square(aDerY.v1())))) ,
               aImGXY.out()
          );
         Tiff_Im::CreateFromIm(aImGX,"/home/mpierrot/Data/COUR/DerX_" + ToString(aK) +".tif");
         Tiff_Im::CreateFromIm(aImGY,"/home/mpierrot/Data/COUR/DerY_" + ToString(aK) +".tif");
         Tiff_Im::CreateFromIm(aImGXY,"/home/mpierrot/Data/COUR/DerModule_" + ToString(aK) +".tif");
    }
}





int main(int argc,char ** argv)
{
//     FiltreSimple();


    EffetDeBord();
    return 0;
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
