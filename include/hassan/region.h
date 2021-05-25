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
#ifndef _HASSAN_REGION_H
#define _HASSAN_REGION_H


///////////////////////////////////////////////////////////////////////////////
/************* isolement des regions **************/

class Isol_region
{
    public:
      Im2D_U_INT1   im_src;
      Im2D<INT,INT> im_reg;
      
      Isol_region(
                   Im2D_U_INT1   Im_src,
                   Im2D<INT,INT> Im_reg,
                   INT           Nb_min_reg,
                   REAL4         Param_der,
                   REAL4         S_grad_min,
                   REAL4         S_grad_max,
                   REAL4         sbas,
                   REAL4         shaut,
		   Output        wgr,
		   INT           nb_colour=8,
                   bool          hyst=true,
                   INT           Nb_min_cont = 10
                 );
                 
      Isol_region(
                   Im2D_U_INT1   Im_src,
                   REAL4         Param_der,
                   REAL4         sbas,
                   REAL4         shaut,
		   Output        wgr,
		   INT           nb_colour=8,
                   bool          hyst=true,
                   INT           Nb_min_cont = 10
                 );

       void xhyster(REAL4 sbas=10, REAL4 shaut=20);
       Im2D_REAL4  get_norm(){return norm;}
       Im2D_REAL4  get_teta(){return teta;}
       Im2D_U_INT1 get_cont(){return cont;}
       ElList<Pt2di> get_list_region(){return l;}

    private:
       Im2D_REAL4    gradx;
       Im2D_REAL4    grady;
       Im2D_REAL4    norm;
       Im2D_REAL4    teta;
       Im2D_U_INT1   m_loc;
       Im2D_U_INT1   cont;
       Output        _wgr;
       ElList<Pt2di>   l;
       REAL4         param_der;
       REAL4         s_grad_min;
       REAL4         s_grad_max;
       REAL4         s_grad; 
       int           nb_min_reg;
       int           nb_reg;
       int           nb_pt_reg;
       int           no_reg;
       int           nc;
       int           nl;
       int           levmax;
       ElList<Pt2di>   lqueu;
       int           nb_min_cont;
       
       void dilate_reg();
       Liste_Pts_INT2  dilate_reg(Flux_Pts flx);

       void max_loc();
       void hyster(REAL4,REAL4);
       void cal_teta();
       void extract_reg();   
       void parcour(int x, int y);
       void deparcour(int x, int y);   
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class Hregion;
class Data_reg:public RC_Object
{
     friend class Hregion;
     
     public:
     private:
    
          Data_reg(Liste_Pts_U_INT2 reg, INT etiq, INT nr);
          virtual ~Data_reg(){}

         Liste_Pts_U_INT2  _reg;
         INT               _etiq;
         INT               _nr;
         REAL              _w; 
         ElList<Hregion>      _lr;
         REAL *            _coef;
         INT               _nb;
         INT               _capa;
         INT               _z;
         REAL              _zl0;
         REAL              _dz;
         bool              _visite;
         void              set_capa(INT nb);
         void              push(REAL);
         void              push(Hregion);
};

/////////////////////////////////////////////////////////////////////////////////////////////
class Hregion : public  PRC0
{
     public :
     
           Hregion ():PRC0(0){}
           Hregion (Liste_Pts_U_INT2 reg, INT etiq, INT nr);

//           Hregion* ptr();

           INT   capa();
           INT   z();  
           bool  visite();
           INT   etiq();
           REAL  zl0();
           REAL  dz();
           ElList<Hregion> lrv();
           void  push(REAL);
           void  push(Hregion);
           void  set_v(bool);
           void  set_z(INT);
           void  set_zl0(REAL);
           void  set_dz(REAL);
           void  set_capa(INT);
           REAL  & operator [](INT i);
           INT   nr();
           REAL  poid();
           void  cal_poid();
           Liste_Pts_U_INT2  rg();
           REAL  cout(INT, REAL);

    private :
     
        inline Data_reg* dtd(){ return (Data_reg*) _ptr;}
};


#endif // _HASSAN_REGION_H

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
