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



#ifndef ELISE_NEIGHBOOR_REL
#define ELISE_NEIGHBOOR_REL


class Data_Neighbourood  :  public RC_Object
{
    friend  class Neighbourhood;
    friend  class Neigh_Rel_Compute;
    friend  class Simple_Neigh_Rel_Comp;

    public :
           inline INT nb_neigh() const {return _nb_neigh;}
           inline INT ** coord() const {return _coord;}
        
    private :

        INT _dim;
        INT _nb_neigh;
                            // the point {0,0,0 ...} is added as a ``hidden'' neigboor,
                            // this is usefull for some internal computation
        INT ** _coord;      // coord[_dim][_nb_neigh +1]
        INT ** _tr_coord;   // coord[_nb_neigh +1][_dim]




        virtual ~Data_Neighbourood();

        void init_tr(); //  init _tr_coord from _coord + init the hidden neighboor

        void init(INT dim,INT nb);
        Data_Neighbourood(Im2D<INT4,INT>);
        Data_Neighbourood(Pt2di *,INT nb);
};


class Arg_Neigh_Rel_Comp
{
      public :
         Arg_Neigh_Rel_Comp(Flux_Pts_Computed *, bool reflexif);
         inline Flux_Pts_Computed * flux() const {return _flux;}
         inline bool  reflexif() const {return _reflexif;}

     private :
         Flux_Pts_Computed * _flux;
         bool                _reflexif;
};

class Neigh_Rel_Compute : public Mcheck
{
   public :

     virtual void set_reflexif(bool) = 0;
     virtual const Pack_Of_Pts *  
             neigh_in_num_dir(  const Pack_Of_Pts *,
                                char ** _is_neigh,
                                INT &   num_dir) = 0;
     Neigh_Rel_Compute(const Arg_Neigh_Rel_Comp &,
                       Data_Neighbourood *,
                       Pack_Of_Pts::type_pack,
                       INT Sz_buf);

     inline Pack_Of_Pts::type_pack type_pack() const { return _type_pack;}
     inline INT nb_neigh() const { return _neigh->_nb_neigh;}

     inline Data_Neighbourood   * neigh() const { return _neigh;}
     inline INT sz_buf() const {return _sz_buf;}

     virtual ~Neigh_Rel_Compute(); 


   protected :

       Pack_Of_Pts            * _pack;
       Data_Neighbourood      * _neigh;
       Pack_Of_Pts::type_pack   _type_pack;

   private :

       
       INT          _sz_buf;
};


class  Neigh_Rel_Not_Comp : public RC_Object
{
     public :
         virtual Neigh_Rel_Compute * 
               compute(const Arg_Neigh_Rel_Comp &) = 0;
     protected :
         Data_Neighbourood * _neigh;
};


#endif //! ELISE_NEIGHBOOR_REL

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
