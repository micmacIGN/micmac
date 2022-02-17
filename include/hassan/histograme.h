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
#ifndef _HASSAN_HISTOGRAME_H
#define _HASSAN_HISTOGRAME_H



//////////////////////////////////////////////////////////////////////////////

class Histograme
{
      Output _out;
      INT    _col_fon;
      INT    _col_line;

      INT    _tx;
      INT    _ty;
      REAL8  _v_max;
      REAL8  _v_min;

    public:

      Histograme(Output out, INT color,  INT tx = 100, INT ty = 100, REAL8 v_max = 1, REAL v_min = 0)
           : _out(out), _col_fon(color/2), _col_line(color), _tx(tx), _ty(ty), _v_max(v_max), _v_min(v_min)
      {
         dessin_grille();
      }

      void dessin_grille(INT nb = 10);
      void dessin(Im1D_REAL8 hist, bool relatif = true);
      void dessin(REAL8* hist, INT tx = 1, bool relatif = true);
      void dessin_accumulatif(REAL8* hist, INT tx = 1, bool relatif = true);

      void set_tx_ty(INT tx, INT ty){_tx = tx; _ty = ty;}
      void set_v_max(REAL8 v_max){_v_max = v_max;}
      void set_v_min(REAL8 v_min){_v_min = v_min;}
      void set_col_line(INT col){_col_line = col;}

      REAL8 v_min(){return _v_min;}
      REAL8 v_max(){return _v_max;}
      
};


extern void lisse_histo(Im1D_REAL8 hist, INT semi_fenet);
extern void dessin_histo(Im1D_REAL8 hist, Output out, INT coulor,  INT x, INT y, bool grille = true);
extern void dessin_histo(Im1D_REAL8 hist, Output out, INT coulor,  INT x, INT y, REAL val_max, bool grille = true);

extern void dessin_histo(Im1D_REAL8 hist, Output out, INT coulor,  INT x, INT y, REAL val_max, REAL val_min, bool grille = true);

extern void dessin_diagrame(ElFilo<Pt2di>& f_p, Output out, INT color, INT color_grille, INT tx, INT ty, bool grille = true);


#endif // _HASSAN_HISTOGRAME_H

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
