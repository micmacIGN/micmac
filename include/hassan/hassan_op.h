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
#ifndef _ELISE_GENERAL_HASSAN_OP_H
#define _ELISE_GENERAL_HASSAN_OP_H


/////////////////////////////////////////////////////////////////////////////
/*    correlation  entre deux images                                       */
/***************************************************************************/

class H_Oper_Correl_Im : public Simple_OPBuf1<REAL,REAL>
{
   public :
        H_Oper_Correl_Im (){}

   private :
       void  calc_buf
                     (
                           REAL ** output,
                           REAL *** input
                     );
};


/////////////////////////////////////////////////////////////////////////////
/*    correlation avec une contriante de region de l'image gauche          */
/***************************************************************************/

class H_Oper_Correl : public Simple_OPBuf1<REAL,REAL>
{
   public :
        H_Oper_Correl (){}
     
   private :
       void  calc_buf
                     (
                           REAL ** output,
                           REAL *** input
                     );
};

/////////////////////////////////////////////////////////////////////////////
/*    correlation avec une contriante de region de l'image gauche          */
/***************************************************************************/

class H_Oper_Correl_Cont : public Simple_OPBuf1<REAL,REAL>
{
   public :
        H_Oper_Correl_Cont (){}
     
   private :
       void  calc_buf
                     (
                           REAL ** output,
                           REAL *** input
                     );
};

/////////////////////////////////////////////////////////////////////////////
/*    filtrage selon les moindres carrees avec une contrainte de region    */
/***************************************************************************/

class H_Oper_Md_Car : public Simple_OPBuf1<REAL,REAL>
{
   public :
        H_Oper_Md_Car (INT param) : _param(param),_tab (0){}
	virtual ~H_Oper_Md_Car();
        virtual  Simple_OPBuf1<REAL,REAL> * dup_comp();   
     
   private :
       void  calc_buf
                     (
                           REAL ** output,
                           REAL *** input
                     );
	INT _param;
        REAL  *_tab;
};  


////////////////////////////////////////////////////////////
/*    Moindre carre                                       */
/**********************************************************/

class Moind_Car
{
   public:
      Moind_Car(INT param = 1);
      ~Moind_Car();

      void push(REAL * v_point, REAL poid=1);      
      void push(Pt3dr p, REAL poid=1);     
      void enlev(Pt3dr p, REAL poid=1);
      void mis_en_zero();
      bool calc();
      bool statu();
      REAL get_param(INT i);
      REAL get_z(Pt2dr p);
      REAL get_z(Pt3dr p);
      REAL get_mq();

   private:
      REAL ** _tab;
      REAL *  _buf;
      REAL *  _prm;
      REAL *  _vect;
      REAL    _poid;
      REAL    _residu;
      bool    _statu;
      INT     _param; 
      INT     _n; 
      void    ajout();  
      void    vect(Pt3dr p, REAL poid);    
};



///////////////////////////////////////////////////////////////////////////////////////////////////
/*    filtrage selon les moindres carrees avec une contrainte de region et poid de coef de corr  */
/*************************************************************************************************/

class H_Oper_Md_Car_Poid : public Simple_OPBuf1<REAL,REAL>
{
   public :
        H_Oper_Md_Car_Poid (INT param):_param(param),_mc(param){}
     
   private :
       void  calc_buf
                     (
                           REAL ** output,
                           REAL *** input
                     );
      INT _param;
      Moind_Car _mc;
};  



#endif  //_ELISE_GENERAL_HASSAN_OP_H

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
