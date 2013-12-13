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



#ifndef  _ELISE_OUT_PUT_H
#define  _ELISE_OUT_PUT_H



class Output_Computed : public Mcheck
{
      public :
           virtual void update
           (
                const Pack_Of_Pts * pts,
                const Pack_Of_Pts * vals
           ) = 0;
           inline INT    dim_consumed(){ return _dim_consumed;}

           virtual ~Output_Computed();
      protected :

           Output_Computed(INT  dim_consumed);

      private :
          INT          _dim_consumed;
};



class Arg_Output_Comp
{
      public :

         Arg_Output_Comp(Flux_Pts_Computed *,Fonc_Num_Computed *);
         inline Flux_Pts_Computed * flux() const {return _flux;}
         inline Fonc_Num_Computed * fonc() const {return _fonc;}

     private :
         Flux_Pts_Computed * _flux;
         Fonc_Num_Computed * _fonc;
};




class Output_Not_Comp : public RC_Object
{
      public :
         virtual  Output_Computed * compute(const Arg_Output_Comp &) = 0;
};


extern Output_Computed * out_adapt_type_fonc
                  (    const Arg_Output_Comp & arg,
                       Output_Computed       * o,
                       Pack_Of_Pts::type_pack type_wished
                  );

extern Output_Computed * out_adapt_type_pts
                  (    const Arg_Output_Comp & arg,
                       Output_Computed       * o,
                       Pack_Of_Pts::type_pack type_wished
                  );



extern Output_Computed * clip_out_put
                 (
                       Output_Computed * o,
                       const Arg_Output_Comp & arg,
                       const INT * p0,
                       const INT * p1
                  );

extern Output_Computed * clip_out_put
                 (
                       Output_Computed * o,
                       const Arg_Output_Comp & arg,
                       Pt2di                    p0,
                       Pt2di                    p1
                  );

   // this output is not really destinated to users;

// Output  push_values (Std_Pack_Of_Pts_Gen **);
template<class Type> Output  push_values (Std_Pack_Of_Pts<Type > ** pp);



#endif //  _ELISE_OUT_PUT_H

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
