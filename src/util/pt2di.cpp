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

bool BadValue(const double & aVal)
{
    return BadNumber(aVal);
}

bool BadValue(const Pt2dr & aP)
{
   return BadNumber(aP.x) || BadNumber(aP.y);
}

bool BadValue(const Pt3dr & aP)
{
   return BadNumber(aP.x) || BadNumber(aP.y) || BadNumber(aP.z);
}

bool BadValue(const ElRotation3D & aR)
{
   if (BadValue(aR.tr())) return true;

   for (int aK=0 ; aK<3 ; aK++)
   {
      Pt3dr aCol;
      aR.Mat().GetCol(aK,aCol);
      if (BadValue(aCol)) return true;
   }

   return false;
}


double cos(int x) {return cos(double(x));}
double sin(int x) {return sin(double(x));}

ostream & operator << (ostream & ofs,const Pt2dr  &p)
{
      ofs << "[" << p.x << "," << p.y <<"]";
      return ofs;
}
ostream & operator << (ostream & ofs,const Pt2di  &p)
{
      ofs << "[" << p.x << "," << p.y <<"]";
      return ofs;
}                        
ostream & operator << (ostream & ofs,const Pt2df  &p)
{
      ofs << "[" << p.x << "," << p.y <<"]";
      return ofs;
}                        

ostream & operator << (ostream & ofs,const Pt2dUi2  &p)
{
      ofs << "[" << p.x << "," << p.y <<"]";
      return ofs;
}                        


ostream & operator << (ostream & ofs,const Pt3dr  &aPt)
{
      ofs << "[" << aPt.x << "," << aPt.y  << "," << aPt.z  <<"]";
      return ofs;
}
ostream & operator << (ostream & ofs,const Pt3di  &p)
{
      ofs << "[" << p.x << "," << p.y  << "," << p.z  <<"]";
      return ofs;
}                        

//== Test SVN 
// Test SVN  2


/**********************************************************************/



template <class Type>   void Pt2d<Type>::to_tab(Type (& t)[2]) const 
{
   t[0] = x; 
   t[1] = y;
}
template <class Type> Pt2d<Type> Pt2d<Type>::FromTab(const Type * aV)
{ 
   return  Pt2d<Type>(aV[0],aV[1]);
}
template <class Type> Pt2d<Type> Pt2d<Type>::FromTab(const std::vector<Type> & aV)
{
    return FromTab(&(aV[0]));
}


template <class Type>   std::vector<Type> Pt2d<Type>::ToTab() const
{
    std::vector<Type> aV;
    aV.push_back(x);
    aV.push_back(y);

    return aV;
}







REAL  average_euclid_line_seed (Pt2di p)
{
      return euclid(p) / dist8(p);
}

template <class Type>  Pt2d<Type>  Pt2d<Type>::Square() const
{
	return  Pt2d<Type>(x*x-y*y,2*x*y);
}


template <class Type> void  Pt2d<Type>::Verif_adr_xy()
{
    El_Internal.ElAssert
    (
         &y == &x+1,
         EEM0 << "Bad assumpution in  Pt2d<Type>::Verif_adr_xy"
    );
}
template <class Type>  Output  Pt2d<Type>::sigma()
{
      Verif_adr_xy();
      return ::sigma(&x,2);
}
template <class Type>  Output  Pt2d<Type>::VMax()
{
      Verif_adr_xy();
      return ::VMax(&x,2);
}
template <class Type>  Output  Pt2d<Type>::VMin()
{
      Verif_adr_xy();
      return ::VMin(&x,2);
}

template <class Type>  Output  Pt2d<Type>::WhichMax()
{
      Verif_adr_xy();
      return ::WhichMax(&x,2);
}
template <class Type>  Output  Pt2d<Type>::WhichMin()
{
      Verif_adr_xy();
      return ::WhichMin(&x,2);
}





Pt2di  best_4_approx(const Pt2di & p)
{
      return (ElAbs(p.x) > ElAbs(p.y))                 ?
             (   (p.x>0) ? Pt2di(1,0) : Pt2di(-1,0)) :
             (   (p.y>0) ? Pt2di(0,1) : Pt2di(0,-1)) ;
}


Pt2di second_freeman_approx(Pt2di u, bool conx_8,Pt2di u1)
{
     Pt2di u2;

     if(u1^u)
     {
        u2 = best_4_approx(u - u1 * scal(u,u1));  // first suppose not 4 cone xity
        if (conx_8)  // eventually correct if 8 conne xity
           u2 = u1+u2;
      }
      // if u is horizontal or vertical (wich is equivalent to u colinear to u1)
     //  the previous line mail fail (because take best_4_approx of (0,0) may give
     // u1). So simply set :
      else
         u2 = u1;

   return u2;
}


INT    num_4_freeman(Pt2di p)
{
    ASSERT_INTERNAL(dist4(p) == 1, "incoherence in num_4_freeman");

    return p.x ? 1-p.x : 2-p.y;
}

Pt2di corner_box_included(Pt2di pmin,Pt2di pmax,bool left,bool down)
{
    return Pt2di
           (
               left  ? pmin.x : pmax.x -1,
               down  ? pmin.y : pmax.y -1
           );
}

REAL  angle(const Pt2dr & p)
{
       assert_not_nul(p);
       return atan2(p.y,p.x);
}

REAL  angle(const Pt2dr & p1,const Pt2dr & p2)
{
      return angle(p2*p1.conj());
}

REAL angle_de_droite(const Pt2dr & p)
{
     return (p.x > 0 ) ? angle(p) : angle (-p);
}

REAL  angle_de_droite(const Pt2dr & p1,const Pt2dr & p2)
{
      return angle_de_droite(p2*p1.conj());
}

REAL angle_de_droite_nor(const Pt2dr & p)
{
     return ElAbs(angle_de_droite(p));
}

REAL  angle_de_droite_nor(const Pt2dr & p1,const Pt2dr & p2)
{
      return angle_de_droite_nor(p2*p1.conj());
}

// Pt2dr polar(const Pt2dr & p,REAL AngDef)
// {
//     if ((p.x==0) && (p.y== 0))
//        return Pt2dr(0,AngDef);
//     return Pt2dr(hypot(p.x,p.y),atan2(p.y,p.x));
// }

template<class Type>  bool
      Pt2d<Type>::in_sect_angulaire(const Pt2d<Type> & p1,const Pt2d<Type> & p2) const
{
    if ((p1^p2) >0)
       return ((p1^*this)>0) && ((*this^p2)>0);
    else
       return ((p1^*this)>0) || ((*this^p2)>0);
}

template<> Pt2d<Fonc_Num>::TypeScalReel Pt2d<Fonc_Num>::RatioMin(const Pt2d<Fonc_Num> & p) const
{
    return Min ( x/(TypeScalReel)p.x, y/(TypeScalReel)p.y);
}


template<class Type> typename Pt2d<Type>::TypeScalReel Pt2d<Type>::RatioMin(const Pt2d<Type> & p) const
{
         return ElMin 
		 ( 
		     x/(typename Pt2d<Type>::TypeScalReel)p.x, 
		     y/(typename Pt2d<Type>::TypeScalReel)p.y
                 );
}

template <> Pt2d<Fonc_Num>::Pt2d(const Pt2d<double>& p) : x (p.x), y (p.y) {};



template <>  bool  Pt2d<Fonc_Num>::in_sect_angulaire(const Pt2d<Fonc_Num> & p1,const Pt2d<Fonc_Num> & p2) const
{
   ELISE_ASSERT(false,"No Pt2d<Fonc_Num>::in_sect_angulaire");
   return false;
}


template <>  void ElSetMax (Fonc_Num & v1,Fonc_Num v2)
{
   ELISE_ASSERT(false,"No ElSetMax (Fonc_Num & v1,Fonc_Num v2)");
}
template <>  void ElSetMin (Fonc_Num & v1,Fonc_Num v2)
{
   ELISE_ASSERT(false,"No ElSetMin (Fonc_Num & v1,Fonc_Num v2)");
}

Fonc_Num operator += (Fonc_Num &,const Fonc_Num &)
{
   ELISE_ASSERT(false,"No operator +=  (Fonc_Num & v1,Fonc_Num v2)");
   return FX+FY;
}

Fonc_Num operator -= (Fonc_Num &,const Fonc_Num &)
{
   ELISE_ASSERT(false,"No operator -=  (Fonc_Num & v1,Fonc_Num v2)");
   return FX+FY;
}

Pt2d<Fonc_Num> operator * (Pt2d<Fonc_Num> aP,Fonc_Num aScal)
{
   return Pt2d<Fonc_Num>(aP.x*aScal,aP.y*aScal);
}

template <>  void set_min_max (Fonc_Num & v1,Fonc_Num & v2)
{
   ELISE_ASSERT(false,"No set_min_max (Fonc_Num & v1,Fonc_Num v2)");
}


#define  DEFOUT(aType)\
template <>    Output  sigma(aType *,INT)         \
{\
   ELISE_ASSERT(false,"No sigma(aType * res,INT)");\
   return Output::onul();\
}\
template <>    Output  VMax(aType *,INT)         \
{\
   ELISE_ASSERT(false,"No sigma(VMax * res,INT)");\
   return Output::onul();\
}\
template <>    Output  VMin(aType *,INT)         \
{\
   ELISE_ASSERT(false,"No sigma(VMin * res,INT)");\
   return Output::onul();\
}\
template <>    Output  WhichMin(aType *,INT)         \
{\
   ELISE_ASSERT(false,"No WhichMin(VMin * res,INT)");\
   return Output::onul();\
}\
template <>    Output  WhichMax(aType *,INT)         \
{\
   ELISE_ASSERT(false,"No WhichMax(VMin * res,INT)");\
   return Output::onul();\
}


DEFOUT(Fonc_Num)
DEFOUT(float)

Fonc_Num TCompl<Fonc_Num>::FromC(double aV)
{
  return Fonc_Num(aV);
}

Fonc_Num ElStdTypeScal<Fonc_Num>::RtoT(double aVal)
{
      return Fonc_Num(aVal);
}

Fonc_Num ElStdTypeScal<Fonc_Num>::RTtoT(Fonc_Num aF)
{
      return aF;
}


Fonc_Num ElStdTypeScal<Fonc_Num>::T2R(Fonc_Num aFonc)
{
   return Rconv(aFonc);
}





/*************************************************************/

template <class Type> istream & REadPt(istream & ifs,Pt3d<Type>  &p)
{
	INT c = ifs.get();
	ELISE_ASSERT(c=='[',"REadPt");

	ifs >> p.x;
	c = ifs.get();
	ELISE_ASSERT(c==',',"REadPt");
	ifs >> p.y;
	c = ifs.get();
	ELISE_ASSERT(c==',',"REadPt");
	ifs >> p.z;
	c = ifs.get();
	ELISE_ASSERT(c==']',"REadPt");

	return ifs;
}


istream & operator >> (istream & ifs,Pt3dr  &p)
{
	return  REadPt(ifs,p);
}

istream & operator >> (istream & ifs,Pt3di  &p)
{
	return  REadPt(ifs,p);
}



REAL ToSpherique(const Pt3dr & aP,REAL & rho,REAL & teta,REAL & phi)
{
    rho = euclid(aP);
    ELISE_ASSERT(rho!=0.0,"ToSpherique, rho=0");
    
    phi = asin(ElMin(1.0,ElMax(-1.0,aP.z/rho)));
    if ((aP.x ==0) && (aP.y==0))
       teta = 0;
    else
       teta = atan2(aP.y,aP.x);

    return rho;
}


// CONSTRUCTEUR 
template <class Type> Pt3d<Type>::Pt3d(const Pt3d<Type> & p)
{
    x = (Type) p.x;
    y = (Type) p.y;
    z = (Type) p.z;
}

template <> Pt3d<int>::Pt3d(const Pt3d<REAL> & p)
{
    // A Prioi j'aurai dit round_ni, mais cas marche comme ca ....
    x = (int) p.x;
    y = (int) p.y;
    z = (int) p.z;
}

template <> Pt3d<double>::Pt3d(const Pt3d<int> & p)
{
    // A Prioi j'aurai dit round_ni, mais cas marche comme ca ....
    x =  p.x;
    y =  p.y;
    z =  p.z;
}




template <class Type> Pt3d<Type>::Pt3d() :
       x (0),
       y (0),
       z (0)
{
}

template <class Type> Pt3d<Type>::Pt3d(Type X,Type Y,Type Z) :
       x (X),
       y (Y),
       z (Z)
{
}

//template Pt3d<REAL16>::Pt3d(REAL16,REAL16,REAL16);

template <class Type> Pt3d<Type>::Pt3d(const Pt2d<Type> & p2,Type Z) :
       x (p2.x),
       y (p2.y),
       z (Z)
{
}

template <class Type> Pt3d<Type>  &  Pt3d<Type>::operator = (const Pt3d<Type> & p2) 
{
    x = p2.x;
    y = p2.y;
    z = p2.z;

    return *this;
}







template <class Type> void  pt_set_min_max(Pt3d<Type> & p0,Pt3d<Type> & p1)
{
     set_min_max(p0.x,p1.x);
     set_min_max(p0.y,p1.y);
     set_min_max(p0.z,p1.z);
}



template <class Type> Pt3d<Type> Pt3d<Type>::operator + (const Pt3d<Type> & p2) const
{
    return Pt3d(x+p2.x,y+p2.y,z+p2.z);
}

template <class Type>  Pt3d<Type> Pt3d<Type>:: operator * (Type lambda) const
{
     return Pt3d<Type>(x*lambda,y*lambda,z*lambda);
}

template <class Type>  Pt3d<Type> Pt3d<Type>:: operator / (Type lambda) const
{
     return Pt3d<Type>(x/lambda,y/lambda,z/lambda);
}

template <class Type> Pt3d<Type> Pt3d<Type>::operator - (const Pt3d<Type> & p2) const
{
    return Pt3d<Type>(x-p2.x,y-p2.y,z-p2.z);
}


template <class Type> Pt3d<Type> Pt3d<Type>::operator - () const
{
    return Pt3d<Type>(-x,-y,-z);
}



template <class Type> Pt3d<Type>  Pt3d<Type>::operator ^ (const Pt3d<Type> & p2) const
{
    return Pt3d<Type>
           (
                y * p2.z - z * p2.y,
                z * p2.x - x * p2.z,
                x * p2.y - y * p2.x
           );
}

template <class Type >
         void  corner_boxes
                 (
                    Pt3d<Type> p1,
                    Pt3d<Type> p2,
                    Pt3d<Type> (& t)[8] 
                  ) 
{
     t[0] =  Pt3d<Type>(p1.x,p1.y,p1.z);
     t[1] =  Pt3d<Type>(p1.x,p1.y,p2.z);
     t[2] =  Pt3d<Type>(p1.x,p2.y,p1.z);
     t[3] =  Pt3d<Type>(p1.x,p2.y,p2.z);
     t[4] =  Pt3d<Type>(p2.x,p1.y,p1.z);
     t[5] =  Pt3d<Type>(p2.x,p1.y,p2.z);
     t[6] =  Pt3d<Type>(p2.x,p2.y,p1.z);
     t[7] =  Pt3d<Type>(p2.x,p2.y,p2.z);
}





template <class Type> void  Pt3d<Type>::Verif_adr_xy()
{
    El_Internal.ElAssert
    (
         (&y == &x+1) && (&z == &y+1),
         EEM0 << "Bad assumpution in  Pt3d<Type>::Verif_adr_xy"
    );
}
template <class Type>  Output  Pt3d<Type>::sigma()
{
      Verif_adr_xy();
      return ::sigma(&x,3);
}

template <class Type>  Output  Pt3d<Type>::VMax()
{
      Verif_adr_xy();
      return ::VMax(&x,3);
}
template <class Type>  Output  Pt3d<Type>::VMin()
{
      Verif_adr_xy();
      return ::VMin(&x,3);
}

template <class Type>  Output  Pt3d<Type>::WhichMax()
{
      Verif_adr_xy();
      return ::WhichMax(&x,3);
}
template <class Type>  Output  Pt3d<Type>::WhichMin()
{
      Verif_adr_xy();
      return ::WhichMin(&x,3);
}




template <class Type>   void Pt3d<Type>::to_tab(Type (& t)[3]) const 
{
   t[0] = x; 
   t[1] = y;
   t[2] = z;
}


template <class Type> Pt3d<Type> Pt3d<Type>::FromTab(const Type * aV)
{ 
   return  Pt3d<Type>(aV[0],aV[1],aV[2]);
}
template <class Type> Pt3d<Type> Pt3d<Type>::FromTab(const std::vector<Type> & aV)
{
    return FromTab(&(aV[0]));
}

template <class Type>   std::vector<Type> Pt3d<Type>::ToTab() const
{
    std::vector<Type> aV;
    aV.push_back(x);
    aV.push_back(y);
    aV.push_back(z);

    return aV;
}


template <class Type> Type  Pt3d<Type>::instantiate()
{
   Pt2d<Type> p2d;
   Pt3d<Type>(p2d,(Type) 0);
   Pt3d<Type> p1;
   p1.sigma();
   p1.VMax();
   p1.VMin();
   p1.WhichMin();
   p1.WhichMax();
   Pt3d<Type> p2(1,2,3);
   Pt3d<Type> Q=Pt3d<Type>::P3ToThisT(Pt3d<INT>());
   Pt3d<Type> R= Pt3d<Type>::P3ToThisT(Pt3d<REAL>());
   Q+R;
   pt_set_min_max(p1,p2);
   p1/2+p2*4;
   -p1-p2;
   scal(p1,p2);
   p1 ^ p2;
   Pt3d<Type> t[8];
  corner_boxes(p1,p2,t);
  Type t3[3];
  p2.to_tab(t3);
  euclid(p1);
/*
*/

   return Q.x + R.y;
}

Pt3dr vunit(const Pt3dr & aP)
{
    REAL aD = euclid(aP);
    ELISE_ASSERT(aD!=0,"Dist Nulle un vunit");
    return aP/aD;
}

Pt3dr OneDirOrtho(const Pt3dr & aP)
{
   Pt3dr aPI =  aP ^ Pt3dr(1,0,0);
   Pt3dr aPJ =  aP ^ Pt3dr(0,1,0);

   const Pt3dr & aRes = (square_euclid(aPI) > square_euclid(aPJ)) ? aPI : aPJ;

   return vunit(aRes);
}

/*************************************************************/

template <class Type> Pt4d<Type>::Pt4d(Type X,Type Y,Type Z,Type T)
{
    x = X;
    y = Y;
    z = Z;
    t = T;
}

template <class Type> Pt4d<Type>::Pt4d()
{
     x=y=z=t=0;
}

template <class Type> Type Pt4d<Type>::instantiate()
{
    Pt4d<Type> TT((Type)0,(Type)0,(Type)0,(Type)0);
     Pt4d<Type>();
     return TT.x;
}



/*************************************************************/


template class Pt3d<REAL16>; 

#define INSTATIATE_SWAP_PT2D(Type) \
template class Pt2d<Type>; \
template class Pt3d<Type>; \
template class Pt4d<Type>; 

#if ElTemplateInstantiation
INSTATIATE_SWAP_PT2D(INT);
INSTATIATE_SWAP_PT2D(REAL);
INSTATIATE_SWAP_PT2D(float);
#endif


template class Pt2d<Fonc_Num>;
template class Pt3d<Fonc_Num>;




/*************************************************************/
/*                                                           */
/*          Seg2d                                            */
/*                                                           */
/*************************************************************/

Seg2d::Seg2d(Pt2dr p0,Pt2dr p1) :
    _empty  (false)
{
    _pts[0] = p0;
    _pts[1] = p1;
}

Seg2d::Seg2d(REAL x0,REAL y0,REAL x1,REAL y1) :
    _empty  (false)
{
    _pts[0] = Pt2dr(x0,y0);
    _pts[1] = Pt2dr(x1,y1);
}

Seg2d::Seg2d()      :
    _empty  (true)
{
}


/*
    algo :
       compute the interval [v0 v1] that correpond
       to curvilenear abscisses of the clipped
       segment  (in repair  p0,p1)
*/

Seg2d  Seg2d::clip(Box2dr b,REAL v0,REAL v1,bool IsSeg) const
{
// [0]  an elementary precaution 
     if (_empty)
        return Seg2d();

      if (IsSeg && b.inside(p0()) &&  b.inside(p1()))
         return * this;

// [2] clip with vertical band

     REAL ux = p1().x - p0().x;
     if (ux)
     {
         if (ux > 0)
         {
             v0 = ElMax(v0,(b._p0.x -p0().x)/ux);
             v1 = ElMin(v1,(b._p1.x -p0().x)/ux);
         }
         else
         {
             v0 = ElMax(v0,(b._p1.x -p0().x)/ux);
             v1 = ElMin(v1,(b._p0.x -p0().x)/ux);
         }
     }
     // [2.2]  if vertical segment
     else
     {
         //  [2.2.1] if outside band : empty
         if ((p0().x < b._p0.x) || (p0().x > b._p1.x))
            return Seg2d();
         //  [2.2.2] else nothing to do for now
     }

// [3] clip with horizontal  band
     REAL uy = p1().y - p0().y;

     if (uy)
     {
         if (uy > 0)
         {
             v0 = ElMax(v0,(b._p0.y -p0().y)/uy);
             v1 = ElMin(v1,(b._p1.y -p0().y)/uy);
         }
         else
         {
             v0 = ElMax(v0,(b._p1.y -p0().y)/uy);
             v1 = ElMin(v1,(b._p0.y -p0().y)/uy);
         }
     }
     else
     {
         if ((p0().y < b._p0.y) || (p0().y > b._p1.y))
            return Seg2d();
     }



    if (v0 <= v1)
    {
        Pt2dr u (ux,uy);
        return Seg2d ( p0()+u*v0, p0()+u*v1);
    }
    else
       return Seg2d();

}

Seg2d  Seg2d::clip(Box2dr b) const
{
    return clip(b,0.0,1.0,true);
}

static REAL Many(const Seg2d & aSeg,const Box2dr & aBox)
{
     return 10.0 + 2.0*(aBox.diam() + euclid(aBox.milieu(),aSeg.milieu()))
	           / euclid(aSeg.p0(),aSeg.p1());
}


Seg2d Seg2d::clipDroite(Box2dr aBox) const
{
    REAL Big =Many(*this,aBox);
    return clip(aBox,-Big,Big,false);
}


Seg2d  Seg2d::clip(Box2di b) const
{
    return clip(Box2dr(b._p0,b._p1));
}




REAL Seg2d::AbsiceInterDroiteHoriz(REAL anOrdonnee) const
{
     REAL dY = y1()-y0();
     ELISE_ASSERT(dY!=0,"Horiz Dr in Seg2d::AbsiceInterDroiteHoriz");

     return x0() + ((x1()-x0())/dY) * (anOrdonnee-y0());
}


ElList<Pt2di>  NewLPt2di(Pt2di p)
{
   return ElList<Pt2di>()+p;
}

/**************************************/
/*                                    */
/*     cInterv1D<Type>                */
/*                                    */
/**************************************/

template <class Type> 
cInterv1D<Type>::cInterv1D
(
     const Type & aV0,
     const Type & aV1
)  :
   mV0    (aV0),
   mV1    (ElMax(aV0,aV1)),
   mEmpty (!(mV0<mV1))
{
}
template <class Type> const Type & cInterv1D<Type>::V0() const {return mV0;}
template <class Type> const Type & cInterv1D<Type>::V1() const {return mV1;}
template <class Type> Type cInterv1D<Type>::Larg() const {return mV1-mV0;}

template <class Type> 
cInterv1D<Type> cInterv1D<Type>::Inter(const cInterv1D<Type> & anInterv) const
{
    return  cInterv1D<Type>
	    (
	         ElMax(mV0,anInterv.mV0),
	         ElMin(mV1,anInterv.mV1)
	    );
}

template <class Type> 
cInterv1D<Type> cInterv1D<Type>::Dilate(const cInterv1D<Type> & anInterv) const
{
    return  cInterv1D<Type>
	    (
	         mV0+anInterv.mV0,
	         mV1+anInterv.mV1
	    );
}

template class cInterv1D<int>;


/**************************************/
/*                                    */
/*     cDecoupageInterv1D             */
/*                                    */
/**************************************/

cDecoupageInterv1D::cDecoupageInterv1D
(
     const cInterv1D<int>  & aIntervGlob,
     int aSzMax,
     const cInterv1D<int>  & aSzBord,
     int                     anArrondi
)  :
   mIntervGlob (aIntervGlob),
   mSzBord     (aSzBord),
   mSzMax      (aSzMax),
   mNbInterv   (round_up(mIntervGlob.Larg()/double(mSzMax))),
   mArrondi    (anArrondi)
{
}

int cDecoupageInterv1D::NbInterv() const
{
    return mNbInterv;
}

int cDecoupageInterv1D::KThBorneOut(int aK) const
{
    int aV = mIntervGlob.V0() +  round_ni((mIntervGlob.Larg() * aK )/mNbInterv);
    if ((mArrondi!=1) && (aK!=0)  && (aK!=mNbInterv))
       aV = arrondi_ni(aV,mArrondi);

   return aV;
/*
    return mIntervGlob.V0() 
           +    round_ni((mIntervGlob.Larg() * aK )/mNbInterv);
*/
}

cInterv1D<int> cDecoupageInterv1D::KthIntervOut(int aK) const
{
    return cInterv1D<int>(KThBorneOut(aK),KThBorneOut(aK+1));
}


cInterv1D<int> cDecoupageInterv1D::KthIntervIn
               (
		   int aK,
		   const cInterv1D<int>  & aSzBord
               ) const
{
   return  mIntervGlob.Inter(KthIntervOut(aK).Dilate(aSzBord));
}


cInterv1D<int> cDecoupageInterv1D::KthIntervIn (int aK) const
{
   return  KthIntervIn(aK,mSzBord);
}

const cInterv1D<int> & cDecoupageInterv1D::IGlob() const
{
   return mIntervGlob;
}

const cInterv1D<int> & cDecoupageInterv1D::IBrd() const
{
   return mSzBord ;
}

int cDecoupageInterv1D::LargMaxOut() const
{
    int aRes = 0;
    for (int aK=0 ; aK<mNbInterv ; aK++)
       ElSetMax(aRes,KthIntervOut(aK).Larg());
    return aRes;
}

int cDecoupageInterv1D::LargMaxIn(const cInterv1D<int> & aBrd) const
{
    int aRes = 0;
    for (int aK=0 ; aK<mNbInterv ; aK++)
       ElSetMax(aRes,KthIntervIn(aK,aBrd).Larg());
    return aRes;
}


int cDecoupageInterv1D::LargMaxIn() const
{
	return LargMaxIn(mSzBord);
}

/**************************************/
/*                                    */
/*     cDecoupageInterv2D             */
/*                                    */
/**************************************/

template <class Type> cInterv1D<Type> IntX(const Box2d<Type> & aBox)
{
   return cInterv1D<Type>(aBox._p0.x,aBox._p1.x);
}

template <class Type> cInterv1D<Type> IntY(const Box2d<Type> & aBox)
{
   return cInterv1D<Type>(aBox._p0.y,aBox._p1.y);
}

template <class Type>
Box2d<Type> I2Box
            (
                 const cInterv1D<Type>  & IntervX,
                 const cInterv1D<Type>  & IntervY
	    )
{
	return Box2d<Type>
               (
		    Pt2d<Type>(IntervX.V0(),IntervY.V0()),
		    Pt2d<Type>(IntervX.V1(),IntervY.V1())
	       );
}




cDecoupageInterv2D::cDecoupageInterv2D
(
         const Box2di & aBGlob,
         Pt2di aSzMax,
         const Box2di   & aSzBord,
         int  anArrondi
) :
  mDecX  ( IntX(aBGlob),aSzMax.x,IntX(aSzBord),anArrondi),
  mDecY  ( IntY(aBGlob),aSzMax.y,IntY(aSzBord),anArrondi),
  mNbX   ( mDecX.NbInterv()),
  mSzBrd (aSzBord)
{
}

cDecoupageInterv2D cDecoupageInterv2D::SimpleDec(Pt2di aSz,int aSzMax,int aSzBrd,int anArrondi)
{
    return cDecoupageInterv2D
           (
                  Box2di(Pt2di(0,0),aSz),
                  Pt2di(aSzMax,aSzMax),
                  Box2di(Pt2di(-aSzBrd,-aSzBrd),Pt2di(aSzBrd,aSzBrd)),
                  anArrondi
           );
}

Pt2di  cDecoupageInterv2D::IndexOfKBox(int aKBox) const 
{
    return Pt2di(aKBox%mNbX,aKBox/mNbX);
}

Box2di cDecoupageInterv2D::DilateBox(int aKBox,const Box2di & aBox,int aDil)
{
   int aNbY = mDecY.NbInterv();
   int aKx = aKBox % mNbX;
   int aKy = aKBox / mNbX;

   Pt2di  aP0
          (
               aBox._p0.x - aDil*(aKx!=0),
               aBox._p0.y - aDil*(aKy!=0)
          );
   Pt2di  aP1
          (
               aBox._p1.x + aDil*(aKx!=mNbX-1),
               aBox._p1.y + aDil*(aKy!=aNbY-1)
          );
   return Box2di(aP0,aP1);
}



int cDecoupageInterv2D::NbInterv() const
{
	return mDecX.NbInterv() * mDecY.NbInterv();
}

Box2di  cDecoupageInterv2D::KthIntervOut(int aK) const
{
     return  I2Box
	     (
	            mDecX.KthIntervOut(aK % mNbX),
		    mDecY.KthIntervOut(aK / mNbX)
	     );
}

Box2di  cDecoupageInterv2D::KthIntervIn(int aK,const Box2di & aBrd) const
{
     return  I2Box
	     (
	            mDecX.KthIntervIn(aK%mNbX,IntX(aBrd)),
		    mDecY.KthIntervIn(aK/mNbX,IntY(aBrd))
	     );
}
	    
Box2di  cDecoupageInterv2D::KthIntervIn(int aK) const
{
     return  KthIntervIn(aK,mSzBrd);
}
	
Pt2di   cDecoupageInterv2D::SzMaxOut() const
{
   return Pt2di(mDecX.LargMaxOut(),mDecY.LargMaxOut());
}

Pt2di   cDecoupageInterv2D::SzMaxIn(const Box2di   & aBrd) const
{
   return Pt2di
          (
	      mDecX.LargMaxIn(IntX(aBrd)),
	      mDecY.LargMaxIn(IntY(aBrd))
          );
}

Pt2di   cDecoupageInterv2D::SzMaxIn() const
{
    return SzMaxIn(mSzBrd);
}

int cDecoupageInterv2D::NbX() const { return mNbX; }

/*
template <> Pt2d<double>::Pt2d(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
template <> Pt2d<float>::Pt2d(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
template <> Pt2d<int>::Pt2d(const Pt2d<double>& p) : x (round_ni(p.x)), y (round_ni(p.y)) {};
*/



template <class Type> istream & REadPt(istream & ifs,Pt2d<Type>  &p)
{
	INT c = ifs.get();
	ELISE_ASSERT(c=='[',"REadPt");

	ifs >> p.x;
	c = ifs.get();
	ELISE_ASSERT(c==',',"REadPt");
	ifs >> p.y;
	c = ifs.get();
	ELISE_ASSERT(c==']',"REadPt");

	return ifs;
}


istream & operator >> (istream & ifs,Pt2dr  &p)
{
	return  REadPt(ifs,p);
}

istream & operator >> (istream & ifs,Pt2di  &p)
{
	return  REadPt(ifs,p);
}


std::vector<Pt2di> PointInCouronne(int aD8Min,int aD8Max)
{
   std::vector<Pt2di> aRes;
   Pt2di aP;
   for (aP.x = -aD8Max+1 ; aP.x < aD8Max ; aP.x++)
   {
       for (aP.y = -aD8Max+1 ; aP.y < aD8Max ; aP.y++)
       {
           if (dist8(aP) >= aD8Min)
              aRes.push_back(aP);
       }
   }
   return aRes;
}

std::vector<std::vector<Pt2di> > PointOfCouronnes(const std::vector<int> &Dist,bool AddD4First)
{
   std::vector<std::vector<Pt2di> > aRes;

   std::vector<Pt2di> aR0 = PointInCouronne(0,Dist[0]);


   if (AddD4First)
   {
       std::vector<Pt2di> aR1;
       std::vector<Pt2di> aR2;
       for (int aK=0 ; aK<int(aR0.size()) ; aK++)
       {
           if (dist4(aR0[aK]) < Dist[0])
           {
               aR1.push_back(aR0[aK]);
           }
           else
           {
               aR2.push_back(aR0[aK]);
           }
       }
       aRes.push_back(aR1);
       aRes.push_back(aR2);
   }
   else
   {
       aRes.push_back(aR0);
   }

   for (int aK=0 ; aK< int(Dist.size()-1) ; aK++)
   {
        aRes.push_back(PointInCouronne(Dist[aK],Dist[aK+1]));
   }

   return aRes;
}

std::vector<std::vector<Pt2di> > StdPointOfCouronnes(int aDMax,bool AddD4First)
{
    std::vector<int> aDist;
    for (int aD=2; aD<=aDMax; aD++)
        aDist.push_back(aD);

   return PointOfCouronnes(aDist,AddD4First);
}


class cCmpPtOnX
{
    public : bool operator()(const Pt2dr & aP1,const Pt2dr & aP2) {return aP1.x < aP2.x;}
};
class cCmpPtOnY
{
    public : bool operator()(const Pt2dr & aP1,const Pt2dr & aP2) {return aP1.y < aP2.y;}
};


Pt3dr PMoyFromEchant(const std::vector<Pt3dr> & anEch)
{
    Pt2dr aSomPt(0,0);
    double aSomPds = 0;
    for (int aK=0;aK<int(anEch.size()) ; aK++)
    {
         const Pt3dr & aP = anEch[aK];
         aSomPt.x += aP.x * aP.z;
         aSomPt.y += aP.y * aP.z;
         aSomPds += aP.z;
    }

    return Pt3dr(aSomPt.x/aSomPds,aSomPt.y/aSomPds,aSomPds);
}


std::vector<Pt3dr>  GetDistribRepreBySort(std::vector<Pt2dr> & aVP,const Pt2di & aNbOut,Pt3dr & aPRep)
{
    Pt2dr * aAdr0 = VData(aVP);
    int aNbIn0 = (int)aVP.size();
    std::vector<Pt3dr> aRes;

    cCmpPtOnX aCmpX;
    cCmpPtOnY aCmpY;
    std::sort(aAdr0,aAdr0+aNbIn0,aCmpX);

    for (int aNx=0 ; aNx<aNbOut.x ; aNx++)
    {
         int aX0 = (aNbIn0 * aNx) / aNbOut.x ;
         int aX1 = (aNbIn0 * (aNx+1)) / aNbOut.x ;

         Pt2dr * aAdrX = aAdr0 + aX0;
         int aNbInX = (aX1-aX0);

         std::sort(aAdrX,aAdrX+aNbInX,aCmpY);
         for (int aNy=0 ; aNy<aNbOut.y ; aNy++)
         {
              int aY0 = (aNbInX * aNy) / aNbOut.y ;
              int aY1 = (aNbInX * (aNy+1)) / aNbOut.y ;

              int aNbInY =  aY1 -aY0;
              if (aNbInY)
              {
                   Pt2dr * aAdrY = aAdrX + aY0;
                   Pt2dr aSomP(0,0);
                   for (int aK=0 ; aK<aNbInY ; aK++)
                   {
                       aSomP = aSomP + aAdrY[aK];
                   }
                   aSomP = aSomP / double(aNbInY);
                   aRes.push_back(Pt3dr(aSomP.x,aSomP.y,aNbInY));
              }
         }
    }

    aPRep = PMoyFromEchant(aRes);
   

    return aRes;
}


template <class TypeCont,class TypeRes>  TypeRes  TplGetDistribRepre
                                     (
                                                 Pt3dr & aCdg,
                                                 const TypeCont & aCont,
                                                 const Pt2di & aNb,
                                                 const TypeRes*
                                     )
{
    TypeRes aRes;

     Pt2dr aP0(-1e60,-1e60);
     Pt2dr aP1( 1e60, 1e60);

     for
     (
           typename TypeCont::const_iterator it=aCont.begin();
           it!=aCont.end();
           it++
     )
     {
          Pt2dr aP = *it;
          aP0.SetSup(aP);
          aP1.SetInf(aP);
     }

     Pt2dr aStep = (aP1-aP0).dcbyc(Pt2dr(aNb));;

     Im2D_REAL8 aIX(aNb.x,aNb.y,0.0);
     Im2D_REAL8 aIY(aNb.x,aNb.y,0.0);
     Im2D_REAL8 aIP(aNb.x,aNb.y,0.0);

     TIm2D<REAL8,REAL> aTx(aIX);
     TIm2D<REAL8,REAL> aTy(aIY);
     TIm2D<REAL8,REAL> aTp(aIP);

     for
     (
           typename TypeCont::const_iterator it=aCont.begin();
           it!=aCont.end();
           it++
     )
     {
          Pt2dr aP = *it;
          Pt2di anInd = round_ni((aP-aP0).dcbyc(aStep));
          anInd = Sup(Pt2di(0,0),Inf(aNb-Pt2di(1,1),anInd));

           aTx.add(anInd,aP.x);
           aTy.add(anInd,aP.y);
           aTp.add(anInd,1);
     }

     Pt2dr aPtSom(0,0);
     double aSom = 0;
     Pt2di anInd;
     for(anInd.x=0 ; anInd.x<aNb.x; anInd.x++)
     {
         for(anInd.y=0 ; anInd.y<aNb.y; anInd.y++)
         {
              double aPds = aTp.get(anInd);
              if (aPds>0)
              {
                 Pt2dr aPt (aTx.get(anInd),aTy.get(anInd));

                 aRes.push_back(Pt3dr(aPt.x/aPds,aPt.y/aPds,aPds));
                 aPtSom = aPtSom + aPt;
                 aSom += aPds;
                 // aRes.push_back(Pt3dr(aTx.get(anInd)/aP,aTy.get(anInd)/aP,aP));
              }
         }
     }

     if (aSom>0)
        aCdg = Pt3dr(aPtSom.x/aSom,aPtSom.y/aSom,aSom);
     else
        aCdg = Pt3dr(0,0,0);


     return aRes;
}


std::vector<Pt3dr> GetDistribRepresentative(Pt3dr & aCdg,const std::vector<Pt2dr> & aV,const Pt2di & aNb)
{

    return TplGetDistribRepre(aCdg,aV,aNb,(std::vector<Pt3dr> *)0);
}






namespace std
{
bool operator < (const Pt3di & aP1,const Pt3di & aP2)
{
   if (aP1.x < aP2.x) return true;
   if (aP1.x > aP2.x) return false;

   if (aP1.y < aP2.y) return true;
   if (aP1.y > aP2.y) return false;

   if (aP1.z < aP2.z) return true;
   if (aP1.z > aP2.z) return false;

   return false;
}




};




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement, a l'utilisation, a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systèmes et ou de leurs données et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
