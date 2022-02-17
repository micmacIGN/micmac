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



/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/
/*****                                                                        ****/
/*****                                                                        ****/
/*****           PROJECTION -PHOTOGRAMETRY                                    ****/
/*****                                                                        ****/
/*****                                                                        ****/
/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/

       /***********************************************************************/
       /***********************************************************************/
       /*                                                                     */
       /*               CAVALIERE                                             */
       /*                                                                     */
       /***********************************************************************/
       /***********************************************************************/

              /*************************************************************************/
              /*                                                                       */
              /*         Linear_Proj_Cav_Comp                                          */
              /*                                                                       */
              /*************************************************************************/

class Linear_Proj_Cav_Comp : public  Fonc_Num_Comp_TPL<REAL>
{
      
      public :

          Linear_Proj_Cav_Comp
          (
               const Arg_Fonc_Num_Comp  &,
               Fonc_Num_Computed *      f,
               REAL                     zobs,
               REAL                     steep
          );


          const Pack_Of_Pts * values(const Pack_Of_Pts * pts);
          virtual ~Linear_Proj_Cav_Comp(); 

      private :




          // variable +or- global used during computation

          Fonc_Num_Computed * _f;
          REAL     _steep;
          bool      _neg_st;
          REAL      _zobs;

          // This filter is rather adatpated to 2-images, anyway :
          // 
          //   [1] do not want to make an exception to the fact that
          //   linear filter can be used with 1-d signal
          // 
           
};

Linear_Proj_Cav_Comp::Linear_Proj_Cav_Comp
(
        const Arg_Fonc_Num_Comp  & arg,
        Fonc_Num_Computed *      f,
        REAL                     zobs,
        REAL                     steep
)     :
      Fonc_Num_Comp_TPL<REAL>(arg,2,arg.flux()),
      _f (f)
{
      _neg_st = (steep < 0);
      _steep  =    ElAbs(steep) * arg.flux()->average_dist();
      _zobs   =    zobs;
}

Linear_Proj_Cav_Comp::~Linear_Proj_Cav_Comp()
{
   delete _f;
}


const Pack_Of_Pts * Linear_Proj_Cav_Comp::values(const Pack_Of_Pts * pts)
{

      Std_Pack_Of_Pts<REAL> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<REAL> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));

      // if steep is negative, just reverse the values
      // it will be reversed again at end; 
      // wont need to back-reverse for _nb =0 or _nb =1 (because in this case, 
      // reverse has no effect)

      INT nb = pts->nb();

      _pack_out->set_nb(nb);
      if (! nb)
         return _pack_out;


      REAL * xr = _pack_out->_pts[0];
      REAL * yr = _pack_out->_pts[1];

      INT  p0[2],p1[2];
      pts->kth_pts(p0,0);
      pts->kth_pts(p1,nb-1);

      REAL x0 = p0[0];
      REAL y0 = (pts->dim() == 2) ? p0[1] : 0;
      REAL xlast = p1[0];
      REAL ylast = (pts->dim() == 2) ? p1[1] : 0;
      if (_neg_st)
      {
          ElSwap(x0,xlast);
          ElSwap(y0,ylast);
          vals->auto_reverse();
      }

      REAL * z = vals->_pts[0];

      if (nb == 1)
      {
          xr[0] = x0;
          yr[0] = y0;
          return _pack_out;
      }

     // the couple couple (i_sup,z_sup)  wil be a point
     // such that :
     //      * the ray starting for (i,_zobs) cross  (i_sup,z_sup)
     //      * (i_sup,z_sup) is always over the DTM (after phase 0)
     //   

      REAL z_sup = _zobs;
      INT  i_sup = 0;
      INT  i = 0;

     
         // [0] treat the case where the ray starting form the first point 
         // is lower than z0;

      for( ; (i<nb) && (z_sup<z[i_sup]) ; i++, z_sup+=_steep)
      {
          xr[i] = x0;
          yr[i] = y0;
      }


          // [1] treat the "normal" case 

      for ( ; i<nb && (i_sup<nb-1) ; i++,z_sup+=_steep)
      {
          while ((i_sup<nb-1) && (z_sup-_steep>=z[i_sup+1]))
          {
                i_sup ++;
                z_sup -= _steep;
          }
          // OK : the ray starting from (i,zobs) intersect DTM
          // in the intervall [isup,isup+1]
          if (i_sup<nb-1)
          {
              // lambda = coordinate od intersection (in [0,1])
              REAL lambda  = (z_sup - z[i_sup])/(z[i_sup+1]-z[i_sup]+_steep);

              // pds = barrycentric coeff
              REAL pds =  (i_sup+lambda) / (nb-1);

              xr[i] = (1-pds) * x0 + pds * xlast;
              yr[i] = (1-pds) * y0 + pds * ylast;
          }
          else
          {
              xr[i] = xlast;
              yr[i] = ylast;
          }
      }

      for( ; (i<nb) ; i++)
      {
          xr[i] = xlast;
          yr[i] = ylast;
      }


      // reorder _pack_out, restore vals 
      if (_neg_st)
      {
          vals->auto_reverse();
          _pack_out->auto_reverse();
      }

      return _pack_out;
}
              /*******************************************************************/
              /*                                                                 */
              /*         Linear_Proj_Cav_Not_Comp                                */
              /*                                                                 */
              /*******************************************************************/

class Linear_Proj_Cav_Not_Comp : public Fonc_Num_Not_Comp
{

     public :
          Linear_Proj_Cav_Not_Comp (Fonc_Num,REAL zobs,REAL steep);

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
               Tjs_El_User.ElAssert
               (
                   _steep != 0.0,
                   EEM0 << "function \"proj_cav\" : steep null"
               );
               Fonc_Num_Computed * fc = _f.compute(arg);

               Tjs_El_User.ElAssert
               (
                   fc->idim_out() == 1,
                   EEM0 << "function \"proj_cav\" : "
                        << "dim out of Fonc_Num should be 1\n"
                        << "|     (dim out = "
                        <<  fc->idim_out() << ")"
               );
              
               fc = convert_fonc_num(arg,fc,arg.flux(),Pack_Of_Pts::real);

               return  new Linear_Proj_Cav_Comp(arg,fc,_zobs,_steep);
          }

          Fonc_Num  _f;
          REAL _zobs;
          REAL _steep;

          virtual bool  integral_fonc (bool) const {return false;}

          virtual INT dimf_out() const {return 2;}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }         
          virtual void  show(ostream & os) const
          {
                os << "[Linear Filter]";
          }                  

};


Linear_Proj_Cav_Not_Comp::Linear_Proj_Cav_Not_Comp(Fonc_Num f,REAL zobs,REAL steep) :
           _f     (f    ),
           _zobs  (zobs ),
           _steep (steep)
{
}

Fonc_Num proj_cav(Fonc_Num f,REAL zobs,REAL steep)
{
      return r2d_adapt_filtr_lin
             (
                new Linear_Proj_Cav_Not_Comp(f,zobs,steep),
                "proj_cav"
             );
}



       /***********************************************************************/
       /***********************************************************************/
       /*                                                                     */
       /*               AUTO STEREOGRAMME                                     */
       /*                                                                     */
       /***********************************************************************/
       /***********************************************************************/

              /*************************************************************************/
              /*                                                                       */
              /*         Phas_auto_stero_Comp                                          */
              /*                                                                       */
              /*************************************************************************/

class Phas_auto_stero_Comp : public  Fonc_Num_Comp_TPL<REAL>
{
      
      public :

          Phas_auto_stero_Comp
          (
               const Arg_Fonc_Num_Comp  &,
               Fonc_Num_Computed *      f,
               REAL                     zobs,
               REAL                     steep
          );

          virtual ~Phas_auto_stero_Comp(){delete _f;}

      private :
          const Pack_Of_Pts * values(const Pack_Of_Pts * pts);






          inline REAL alt_interpole(REAL x)
          {
                INT i = round_down(x);
                if (i == x) return _alt[i];
                REAL pds = x-i;
                return  (1-pds) * _alt[i] + pds * _alt[i+1];
          }



          // return the absci of intersection of ray crossing p0 (of slope -_steep)
          // with the plane  z = _zobs; do not care about possible intersection with dtm
          inline REAL intersec_incr(Pt2dr p0)
          {
               return p0.x - (_zobs-p0.y) / _steep;
          }
           


         // In this class, when use Pt2dr : x = coordinate on the line, y = alt
         // given a point p0 so that we are sure that upper part of the ray crossing p0
         // do not intersect the dtm, this function compute the intersection of the lower
	 // part with the dtm; the value of x is <0 is this ray do not exist
         // Pt2dr intersec_desc(Pt2dr p0);
         // Pt2dr *  _inter_desc;
         // REAL  *  _inter_incr;


          Fonc_Num_Computed * _f;
          REAL     _steep;     // always > 0
          REAL     _zobs;


              // + or - global variables
          REAL * _alt;
          INT    _nb;
};

Phas_auto_stero_Comp::Phas_auto_stero_Comp
(
        const Arg_Fonc_Num_Comp  & arg,
        Fonc_Num_Computed *      f,
        REAL                     zobs,
        REAL                     steep
)     :
      Fonc_Num_Comp_TPL<REAL>(arg,1,arg.flux()),
      _f (f),
      _steep (ElAbs(steep)),
      _zobs  (zobs)
{
}



const Pack_Of_Pts * Phas_auto_stero_Comp::values(const Pack_Of_Pts * pts)
{
      _nb =  pts->nb();
      _pack_out->set_nb(_nb);
      if (! _nb)
         return _pack_out;

      Std_Pack_Of_Pts<REAL> * pvals =
                   SAFE_DYNC(Std_Pack_Of_Pts<REAL> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));

      _alt = pvals->_pts[0];
      El_User_Dyn.ElAssert
      (
          OpMax.red_tab(_alt,_nb) < _zobs,
          EEM0 << "function \"phasis_auto_stereogramme\" : \n"
               << "|   DTM  altitude should be bellow  observation altitude\n"
               << "|   observation = " << _zobs
               << ", DTM : " << OpMax.red_tab(_alt,_nb)
      );


      REAL * phas = _pack_out->_pts[0];
      REAL  phas_max = -1;



      for (int x = 0; x < _nb ; x++)
      {
          REAL last_x = x; 
          while (last_x >= 0)  // !! NOT ">0", sinon on rentre jamais pour x = 0
          {
               phas[x] = last_x;
               REAL z = alt_interpole(last_x);     
               last_x =  intersec_incr(Pt2dr(last_x,z));
          }
          phas_max = ElMax(phas_max,phas[x]);
      }

      REAL z = alt_interpole(phas_max);
      phas_max -= intersec_incr(Pt2dr(phas_max,z));

      ASSERT_INTERNAL
      (
          phas_max  > 0,
         "incoherence in Phas_auto_stero_Comp::values"
      );

	  {
		for (int x = 0; x < _nb ; x++)
			  phas[x] /= phas_max;
	  }

      return _pack_out;
}
     


              /*******************************************************************/
              /*                                                                 */
              /*         Phas_auto_stero_Not_Comp                                */
              /*                                                                 */
              /*******************************************************************/

class Phas_auto_stero_Not_Comp : public Fonc_Num_Not_Comp
{

     public :
          Phas_auto_stero_Not_Comp (Fonc_Num,REAL zobs,REAL steep);

     private :


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
               Tjs_El_User.ElAssert
               (
                   _steep != 0.0,
                   EEM0 
                    <<  "function \"phasis_auto_stereogramme\" : steep null"
               );
               Fonc_Num_Computed * fc = _f.compute(arg);

               Tjs_El_User.ElAssert
               (
                   fc->idim_out() == 1,
                   EEM0 << "function \"phasis_auto_stereogramme\" :"
                        << " dim out of Fonc_Num should be 1\n"
                        << "|     (dim out = "
                        <<  fc->idim_out() << ")"
               );
              
               fc = convert_fonc_num(arg,fc,arg.flux(),Pack_Of_Pts::real);

               return  new Phas_auto_stero_Comp(arg,fc,_zobs,_steep);
          }

          Fonc_Num  _f;
          REAL _zobs;
          REAL _steep;

          virtual bool  integral_fonc (bool) const 
          {return false;}

          virtual INT dimf_out() const {return 1;}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const
          {
                os << "[Linear Filter]";
          }                  
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }         
};


Phas_auto_stero_Not_Comp::Phas_auto_stero_Not_Comp(Fonc_Num f,REAL zobs,REAL steep) :
           _f     (f    ),
           _zobs  (zobs ),
           _steep (steep)
{
}

Fonc_Num phasis_auto_stereogramme(Fonc_Num f,REAL zobs,REAL steep)
{
      return r2d_adapt_filtr_lin
             (
                new Phas_auto_stero_Not_Comp(f,zobs,steep),
                "phasis_auto_stereogramme"
             );
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
