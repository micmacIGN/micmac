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


#pragma once

/***********************************************************************/
/*                                                                     */
/*       ImOutNotComp                                                  */
/*                                                                     */
/***********************************************************************/

extern INT  PTS_00000000000000[];

class   ImOutNotComp : public  Output_Not_Comp
{
     public :

          Output_Computed * compute(const Arg_Output_Comp & arg);
          ImOutNotComp
             (DataGenIm * gi,GenIm  pgi,bool auto_clip_rle,bool auto_clip_int);

     protected :
          DataGenIm * _gi;
          GenIm      _pgi;
          bool    _auto_clip_rle;
          bool    _auto_clip_int;
};

class   ImRedAssOutNotComp : public  ImOutNotComp
{
     public :
          ImRedAssOutNotComp
          (  
              const OperAssocMixte & op,
              DataGenIm * gi,
              GenIm  pgi,
              bool   auto_clip
           );

     private :
          Output_Computed * compute(const Arg_Output_Comp & arg);
          const OperAssocMixte & _op;
};

/***********************************************************************/
/*                                                                     */
/*       GenImOutRleComp                                               */
/*                                                                     */
/***********************************************************************/

class GenImOutRLE_Comp : public Output_Computed
{

    protected :

        DataGenIm * _gi;

        GenImOutRLE_Comp(DataGenIm *);
        virtual ~GenImOutRLE_Comp();
};


/***********************************************************************/
/*                                                                     */
/*       ImOutRLE_Comp <Type,TypeIm>                                   */
/*                                                                     */
/***********************************************************************/

template <class TypeIm> class ImOutRLE_Comp :
         public GenImOutRLE_Comp
{
    public :
        virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
        ImOutRLE_Comp(DataGenIm *gi);
        virtual ~ImOutRLE_Comp();
};


/***********************************************************************/
/*                                                                     */
/*       ImInRLE_Comp <TypeBase>                                       */
/*                                                                     */
/***********************************************************************/


template <class TypeBase> class ImInRLE_Comp : 
         public Fonc_Num_Comp_TPL <TypeBase>
{
    public :
        virtual const Pack_Of_Pts * values(const Pack_Of_Pts * vals);
        ImInRLE_Comp
        (
                     const Arg_Fonc_Num_Comp &,
                     Flux_Pts_Computed * flux,
                     DataGenIm             *gi,
                     bool                  with_def_value
        );

    private :
        DataGenIm *              _gi;
        bool                     _with_def_value ;
};


class ImInNotComp : public Fonc_Num_Not_Comp
{

       public :

           virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);
           ImInNotComp(DataGenIm *,GenIm ,bool,REAL def);

       private :
 
             bool                 _with_def_value;
             REAL                 _def_value;
             DataGenIm *          _gi;
             GenIm               _pgi;

             virtual  bool  integral_fonc (bool integral_flux) const ;
             virtual  INT  dimf_out () const ;
             void VarDerNN(ElGrowingSetInd &)const ;

};


/***********************************************************************/
/*                                                                     */
/*       ImOutInteger <TypeBase>                                       */
/*                                                                     */
/***********************************************************************/


template <class TypeBase> class ImOutInteger : public Output_Computed
{
    public :
        ImOutInteger(const Arg_Output_Comp &, DataGenIm *gi,bool auto_clip);

        virtual ~ImOutInteger();
   private :
        virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
        DataGenIm                      *   _gi;
        bool                          _auto_clip;
};



/***********************************************************************/
/*                                                                     */
/*       ImInInteger <TypeBase>                                        */
/*                                                                     */
/***********************************************************************/


template <class TypeBase> class ImInInteger : 
                             public Fonc_Num_Comp_TPL <TypeBase>
{
    public :
        virtual const Pack_Of_Pts * values(const Pack_Of_Pts * pts);
        ImInInteger
        (
                 const Arg_Fonc_Num_Comp &, 
                 DataGenIm *gi,
                 bool              with_def_value 
        );

   private :
        DataGenIm *          _gi;
        bool                 _with_def_value;
};



/***********************************************************************/
/*                                                                     */
/*       ImInReal                                                      */
/*                                                                     */
/***********************************************************************/


class ImInReal : public Fonc_Num_Comp_TPL <REAL>
{
    public :
        virtual const Pack_Of_Pts * values(const Pack_Of_Pts * pts);
        ImInReal
        (
                 const Arg_Fonc_Num_Comp &, 
                 DataGenIm *gi,
                 bool              with_def_value 
        );

   private :
        DataGenIm *          _gi;
        bool                 _with_def_value;
};






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
