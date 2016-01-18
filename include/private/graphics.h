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



#ifndef _ELISE_PRIVATE_GRAPHICS_H
#define _ELISE_PRIVATE_GRAPHICS_H

class Data_Col_Pal : public RC_Object
{
    public :
      friend class Col_Pal;
      friend class Data_Elise_Gra_Disp;  // => Data_Col_Pal()
      friend class PS_Out_RLE_computed;

      INT    num(){return _num;}
      INT    get_index_col(Data_Disp_Set_Of_Pal *);

      inline bool eg_dcp(const Data_Col_Pal & p2)
      {
          return     (_num == p2._num)
                 ||  (
                            (_c[0] == p2._c[0])
                        &&  (_c[1] == p2._c[1])
                        &&  (_c[2] == p2._c[2])
                        &&  (_pal.dep() == p2._pal.dep())
                     );
      }
      Elise_Palette  pal()  {return  _pal;}
      const int *    cols() {return _c;}

    private :
  

      Data_Col_Pal(Elise_Palette,INT,INT = -1,INT = -1);
      Data_Col_Pal();  // initialize an impossible colour

      Elise_Palette         _pal;
      INT                   _c[3];
      INT                   _num;

      static  INT           _number_tot;


};


class Data_Line_St : public RC_Object
{
      friend class Line_St;
      friend class Data_Elise_Gra_Win;

      public  :
        inline REAL witdh() const {return _width;}
        inline Data_Col_Pal * dcp() const {return _col.dcp();}
        inline Col_Pal  col() const {return _col;}

      private :

         Data_Line_St(Col_Pal,REAL witdh);



        Col_Pal _col;
        REAL     _width;
};

class Data_Fill_St : public RC_Object
{
      friend class Fill_St;

      public  :
        inline Data_Col_Pal * dcp() const {return _col.dcp();}
        inline Col_Pal  col() const {return _col;}

      private :


         Data_Fill_St(Col_Pal);
         Col_Pal _col;
};

#endif  // ! _ELISE_PRIVATE_GRAPHICS_H





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
