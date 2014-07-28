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



static const INT NBB = 8;

Pt2dr  prolong
      (
            TIm2D<INT1,INT> & im,
            const Seg2d     & seg,
            REAL              step,
            REAL              Rvlim,
            bool              BorneInf,
            bool              etirement
      )
{
    Pt2dr dec = vunit(seg.p0()-seg.p1()) * step; 
    if (! etirement)
       dec = -dec;
    Pt2dr pt = seg.p0();

    INT Ivlim = (INT)(Rvlim * (1<<(2*NBB)));

    Pt2di p0 (1,1);
    Pt2di p1 (im.sz()-Pt2di(2,2));
    
    for (;;)
    {
        ElPFixed<NBB> Fpt (pt);
        if (! Fpt.inside(p0,p1)) 
           return pt;
        INT  v = TImGet<INT1,INT,NBB>::getb2(im,Fpt); 
        if ( BorneInf   ?  (v>Ivlim)  : (v<Ivlim)  )
           pt += dec;
       else
          return pt;
    }
}


Pt2dr prolong_iter
      (
            TIm2D<INT1,INT> & im,
            const Seg2d   &   seg,
            REAL              step,
            REAL              step_min,
            REAL              Rvlim,
            bool              BorneInf,
            bool              etirement
      )
{
    Pt2dr p0 = seg.p0();
    Pt2dr p1 = seg.p1();
    while (step > step_min)
    {
         p0 =  prolong(im,Seg2d(p0,p1),step,Rvlim,BorneInf,etirement);
         step /= 2.0;
    }
    return p0;
}


Pt2dr  prolongt_std_uni_dir
       (
            Im2D<INT1,INT>   im,
            const Seg2d   &   seg,
            REAL              step,
            REAL              vlim
       )
{
     TIm2D<INT1,INT>   Tim (im);
     return  prolong_iter(Tim,seg,step,0.05,vlim,true,true);
}

Seg2d  prolongt_std_bi_dir
       (
            Im2D<INT1,INT>   im,
            const Seg2d   &   seg,
            REAL              step,
            REAL              vlim
       )
{
     TIm2D<INT1,INT>   Tim (im);
     Pt2dr q0 =   prolong_iter(Tim,seg,step,0.05,vlim,true,true);
     Pt2dr q1 =   prolong_iter(Tim,seg.reverse(),step,0.05,vlim,true,true);
     return Seg2d(q0,q1);
}

Seg2d  retract_std_bi_dir
       (
            Im2D<INT1,INT>   im,
            const Seg2d   &   seg,
            REAL              step,
            REAL              vlim
       )
{
     TIm2D<INT1,INT>   Tim (im);
     Pt2dr q0 =   prolong_iter(Tim,seg,step,0.05,vlim,false,false);
     Pt2dr q1 =   prolong_iter(Tim,seg.reverse(),step,0.05,vlim,false,false);
     return Seg2d(q0,q1);
}

Seg2d  retr_prol_std_bi_dir
       (
            Im2D<INT1,INT>   im,
            const Seg2d   &   seg,
            REAL              step,
            REAL              vlim
       )
{
      Seg2d res =  prolongt_std_bi_dir(im,seg,step,vlim);
      return  retract_std_bi_dir(im,res,step,vlim);
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
