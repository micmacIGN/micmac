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




class SomApproxPoly;
class Approx_poly;
class ArgAPP;


/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/


ArgAPP::ArgAPP
(
    REAL prec,
    INT nb_jump,
    ModeCout mcout,
    ModeSeg mseg,
    bool    freem_sup,
    INT nb_step 
) :
  _prec            (prec),
  _nb_jump         (nb_jump),
  _mcout           (mcout),
  _mseg            (mseg),
  _freem_sup       (freem_sup),
  _nb_step         (nb_step),
  mInitWithEnvConv (false)
{
   if (mcout==DMaxDroite)
   {
       mDMax = _prec;
       _prec = 1.0;  // La precision n'a alors aucune importance tant que > 0
   }
}

bool ArgAPP::InitWithEnvConv() const {return mInitWithEnvConv;}
void ArgAPP::SetInitWithEnvConv() {mInitWithEnvConv=true;}

/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/


template <class T> INT get_best_intexe(const ElFifo<T> & fp,bool EnvConv)
{
   if (! fp.circ())
      return 0;

   if (fp.nb() < 4)
      return 0;


   INT delta = EnvConv ? 1 : ElMin(5,(fp.nb()-2)/2);
   REAL min_cos = 10.0;
   INT best_index = 0;

   std::vector<INT> aVOk;
   std::vector<INT> aVPrec;
   std::vector<INT> aVSucc;

   for(INT aK=0 ; aK<INT(fp.size()) ; aK++)
   {
       aVOk.push_back(EnvConv ? 0 : 1);
       aVPrec.push_back(aK-delta);
       aVSucc.push_back(aK+delta);
   }
   if (EnvConv)
   {
      ElFilo<Pt2dr> aFLP;
      for(INT aK=0 ; aK<INT(fp.size()) ; aK++)
         aFLP.pushlast(fp[aK]);
      ElFifo<INT> Ind;
      env_conv(Ind,aFLP,true);
      for (INT aK=0 ; aK<Ind.nb() ; aK++)
      {
          aVOk[Ind[aK]] = 1;
          aVPrec[Ind[aK]] = Ind[(aK-1+Ind.nb())%Ind.nb()];
          aVSucc[Ind[aK]] = Ind[(aK+1)%Ind.nb()];
      }
   }

   for (INT k =0 ; k<fp.nb() ; k++)
   {
       if (aVOk[k])
       {
            T u1 = fp[k]-fp[aVPrec[k]];
            T u2 = fp[aVSucc[k]]-fp[k];
            double d1 = euclid(u1);
            double d2 = euclid(u2);
            if (d1 && d2)
            {
               double cosin = scal(u1,u2) / (d1*d2);
               if (cosin < min_cos)
               {
                  min_cos = cosin; 
                  best_index = k;
               }
            }
       }
   }
   return best_index ;
}

/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/

class SomApproxPoly : public Mcheck
{
     friend class Approx_poly;

     public :

        void  init_link (SomApproxPoly *pred);

        RMat_Inertie  inert_dif(SomApproxPoly * succ)
        {
             return succ->_mat -_pred_imme->_mat;
        }

        void set_pt(Pt2dr pt);

        Pt2dr _pt;
        REAL _cost;

        RMat_Inertie  _mat;

        SomApproxPoly * _pred;
        SomApproxPoly * _next;
        SomApproxPoly * _pred_imme;
        SomApproxPoly * _best_anc;
       
};


void SomApproxPoly::init_link (SomApproxPoly * pred )
{
    _next = 0;

    _pred = pred;
    _pred_imme = pred;
    pred->_next = this;
}

void SomApproxPoly::set_pt(Pt2dr pt)
{
   _pt = pt;
   _mat = _pred_imme->_mat.plus_cple(pt.x,pt.y);
}

     //========================================

typedef Seg2d  (* APP_calcul_seg_interp) (SomApproxPoly * s1,SomApproxPoly * s2);

Seg2d  APP_seg_extre(SomApproxPoly * s1, SomApproxPoly  *s2)
{
    return Seg2d (s1->_pt,s2->_pt);
}

Seg2d  APP_seg_mean_square(SomApproxPoly * s1, SomApproxPoly  *s2)
{
    RMat_Inertie   m = s1->inert_dif(s2);
    Seg2d         s = seg_mean_square(m);
    SegComp       sc(s.p0(),s.p1());
    return  Seg2d(sc.proj_ortho_droite(s1->_pt),sc.proj_ortho_droite(s2->_pt));
}




     //========================================

typedef double (* APP_calcul_cout)(SomApproxPoly * s1,SomApproxPoly * s2,const SegComp&,const ArgAPP &);

double APP_cout_square_droite(SomApproxPoly * s1,SomApproxPoly * s2,const SegComp& seg,const ArgAPP &)
{
   RMat_Inertie   m = s1->inert_dif(s2);
   double d        =  square_dist_droite(seg,m);
   return sqrt(ElMax(0.0,d*m.s()));
}



double APP_cout_seuil(SomApproxPoly * s1,SomApproxPoly * s2,const SegComp& seg,const ArgAPP & arg)
{
    ELISE_ASSERT (s1 < s2,"APP_cout_seuil s1>=s2");
    for(SomApproxPoly *s = s1; s<=s2 ; s++)
    {
        if (seg.dist_seg(s->_pt) > arg.mDMax)
           return 1e5;
    }
    return 0;

}

/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/

class Approx_poly
{
     public :

         Approx_poly (const ElFifo<Pt2dr> & fp,ArgAPP arg);

         INT  one_pass_pcc();
         void pcc_until_stab(ElFifo<INT>&);

         ~Approx_poly ();

    private :
         
         bool freem_supprimable(INT k);
         void init(int nb,const ArgAPP & arg);

         SomApproxPoly * last_som();
         SomApproxPoly * prem_som();
         INT index_som(SomApproxPoly *);

         SomApproxPoly *       _s;
         INT                   _nb;
         INT                    _ind0;
         bool                   _circ;
         ArgAPP                _arg;
         APP_calcul_cout        _calc_cout;
         APP_calcul_seg_interp  _calc_seg;
};

SomApproxPoly * Approx_poly::last_som() {return _s+_nb-1;}
SomApproxPoly * Approx_poly::prem_som() {return _s;}
INT Approx_poly::index_som(SomApproxPoly * som) {return (int) (som-_s);}

void  Approx_poly::init(int nb,const ArgAPP & arg)
{
     _s = NEW_VECTEUR(-1,nb,SomApproxPoly);
     _s[-1]._mat = RMat_Inertie();
     _nb = nb;

     for (int i = 0; i<nb ; i++)
         _s[i].init_link(_s+i-1);

     _s[0]._pred = 0;
     _s[0]._best_anc = 0;
     _s[0]._cost = 0.0;

     switch (arg._mcout)
     {
          case ArgAPP::D2_droite :
               _calc_cout = APP_cout_square_droite;
          break;
	  case ArgAPP::DMaxDroite :
               _calc_cout = APP_cout_seuil;
          break;
     };
     switch (arg._mseg)
     {
          case ArgAPP::Extre :
               _calc_seg = APP_seg_extre;
          break;

          case ArgAPP::MeanSquare :
               _calc_seg = APP_seg_mean_square;
          break;
     };
}

Approx_poly::~Approx_poly ()
{
   DELETE_VECTOR(_s,-1);
}

INT  Approx_poly::one_pass_pcc()
{
	SomApproxPoly * som;
     for
     (
        som = prem_som()->_next;
        (som != 0);
        som = som->_next
     )
     {
         SomApproxPoly * ancetre = som->_pred;
         bool found = false;

         for
         (
             int step = 0;
             (ancetre!=0) && (step!= _arg._nb_jump);
             ancetre = ancetre->_pred , step++
         )
         {
              Seg2d  seg = _calc_seg(ancetre,som);
              Pt2dr p0 = seg.p0(); 
              Pt2dr p1 = seg.p1(); 
              if (euclid(p0-p1) < 1e-5)
                 p1 = p1 + Pt2dr(1e-5,1e-5);
              SegComp  SComp (p0,p1);

              REAL cost =   ancetre->_cost
                          + _arg._prec
                          + _calc_cout(ancetre,som,SComp,_arg);
              if ((!found) || (cost <  som->_cost))
              {
                   som->_cost = cost;
                   som->_best_anc = ancetre;
                   found = true;
              }
         }
         El_Internal.ElAssert
         (
             found,
             EEM0 << "Incoherence in approx_polygonale"
         );
     }

     INT res = 1;
     som = last_som();
     for
     (
           SomApproxPoly * pred = som->_best_anc    ;
          (pred != 0)                               ;
          som = pred, pred =pred->_best_anc, res++
     )
     {
           som->_pred = pred;
           pred->_next = som;
     }
     return res;
}

void  Approx_poly::pcc_until_stab(ElFifo<INT> & res)
{
    INT nb = one_pass_pcc();

    for (
              INT nb_last = nb+1, step = 1         ;
              (nb_last != nb) && (step < _arg._nb_step) ;
              step++
        )
   {
        nb_last = nb;
        nb = one_pass_pcc();
    }

    res.clear();
    for
    (
           SomApproxPoly * sommet = last_som()  ;
          (sommet != 0)                         ;
          sommet = sommet->_best_anc
    )
           res.pushfirst(index_som(sommet)+_ind0);
}



bool Approx_poly::freem_supprimable(INT ind)
{
     if (ind <= 1)  return false;

     INT k0 = freeman_code(Pt2di(_s[ind-1]._pt-_s[ind]._pt));
     if (k0<0) return false;
     INT k1 = freeman_code(Pt2di(_s[ind]._pt-_s[ind+1]._pt));
     if (k1<0) return false;
     return k1==k0;
}

Approx_poly::Approx_poly
            (
                  const ElFifo<Pt2dr> & fp,
                  ArgAPP        arg
            )  :
               _arg (arg)
{
      _ind0 = get_best_intexe(fp,arg.InitWithEnvConv());
      _circ = fp.circ();
      init(fp.nb()+_circ,arg);
      for (int i = 0; i<_nb; i++)
          _s[i].set_pt(fp[i+_ind0]);


      if (arg._freem_sup)
      {
          INT nb_sup = 0;
          for 
          (
               INT k0 = _nb-1;
               k0>=2;
               k0--
          )
          {
               INT k1 = k0-1;
               while (freem_supprimable(k1))
               {
                  k1--;
                  _s[k0]._pred = _s+k1;
                  _s[k1]._next = _s+k0;
                  nb_sup++;
               }
          }
      }
}

void approx_poly
     (
        ElFifo<INT> &         res,
        const ElFifo<Pt2dr> & fp,
        ArgAPP                arg
     )
{
     Approx_poly  app(fp,arg);
     app.pcc_until_stab(res);
}

std::vector<int> approx_poly
     (
        const std::vector<Pt2dr> & fp,
	bool                       Circ,
        ArgAPP                arg
     )
{
     ElFifo<INT>  res;
     ElFifo<Pt2dr> aFP(fp,Circ);
     Approx_poly  app(aFP,arg);
     app.pcc_until_stab(res);

     std::vector<int> aRV;
     for (INT aK=0 ; aK< (INT(res.size())-Circ) ; aK++)
         aRV.push_back(res[aK]%INT(fp.size()));
     return aRV;
}




void approx_poly
     (
        ElFifo<INT> &         res,
        const ElFifo<Pt2di> & fpi,
        ArgAPP                arg
     )
{
    ElFifo<Pt2dr> fpr(fpi.size(),fpi.circ());;

    for (INT aK=0 ; aK<INT(fpi.size()) ; aK++)
        fpr.push_back(Pt2dr(fpi[aK]));

    approx_poly(res,fpr,arg);
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
