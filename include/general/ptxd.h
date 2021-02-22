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

#ifndef _ELISE_INCLUDE_GENERAL_PTXD_H_
#define _ELISE_INCLUDE_GENERAL_PTXD_H_


//#define Pt2di  Pt2d<INT>
//#define Pt2dr  Pt2d<REAL>

template <class Type> class Box2d;
class Output;
class SegComp;
class Seg2d;
class cElTriangleComp;
template <class Type> class Pt3d;

inline INT  scal(INT v1 ,INT v2 ) { return v1 * v2;}
inline REAL scal(REAL v1,REAL v2) { return v1 * v2;}




template <class Type> class ElStdTypeScal
{
    public :
        typedef ElStdTypeScal<REAL>  TypeReel;
        typedef Type                 TypeScal;
        typedef Type                 TypeEff;
        typedef bool                 TypeBool;

        typedef  REAL TypeScalReel;
        typedef  Type TypeVarProv;
        typedef  REAL TypeVarProvReel;

        static  Type RtoT(REAL);
        static  Type RTtoT(TypeScalReel);
        static  REAL T2R (const Type & aV) {return aV;}
        static Type  El0 () {return 0;}

    private :
};


ElTmplSpecNull INT   ElStdTypeScal<INT>::RtoT(REAL v); //  { return round_ni(v);}
ElTmplSpecNull REAL  ElStdTypeScal<REAL>::RtoT(REAL v); //  { return v;}
ElTmplSpecNull INT   ElStdTypeScal<INT>::RTtoT(REAL v) ; // { return round_ni(v);}
ElTmplSpecNull REAL  ElStdTypeScal<REAL>::RTtoT(REAL v) ; // { return v;}
/*
*/
// INT totoR() {return ElStdTypeScal<INT>::RtoT(3.0);}
// REAL totoI() {return ElStdTypeScal<REAL>::RtoT(3.0);}



class Fonc_Num;
class Symb_FNum;
// Fonctions a Effets de bords sur Fonc_Num => Erreurs Fatales
template <>  void ElSetMax (Fonc_Num & v1,Fonc_Num v2);
template <>  void ElSetMin (Fonc_Num & v1,Fonc_Num v2);
Fonc_Num operator += (Fonc_Num &,const Fonc_Num &);
Fonc_Num operator -= (Fonc_Num &,const Fonc_Num &);
template <>  void set_min_max (Fonc_Num &,Fonc_Num &);

template <> class ElStdTypeScal<Fonc_Num>
{
    public :
        typedef ElStdTypeScal<Fonc_Num>  TypeReel;
        typedef Fonc_Num                 TypeScal;
        typedef Fonc_Num                 TypeEff;
        typedef Fonc_Num                 TypeBool;

        typedef  Fonc_Num TypeScalReel;
        typedef  Symb_FNum  TypeVarProv;

        typedef  Symb_FNum TypeVarProvReel;

        static  Fonc_Num RtoT(REAL aV);
        static  Fonc_Num RTtoT(Fonc_Num);
        static  Fonc_Num T2R(Fonc_Num aV);
        static Fonc_Num  El0 ();
    private :
};




template <class Type> class Pt2d;

template <class Type>
void pt_set_min_max(Pt2d<Type> & p0,Pt2d<Type> & p1);

template <class Type> class TCompl
{
    public :
        // definition par defaut debile, faite pour Fonc_Num
        typedef double  TypeCompl;
};

template <> class TCompl<Fonc_Num>
{
    public :
    // A priori inutile, pour eviter un overlaod
        typedef double  TypeCompl;
        static Fonc_Num FromC(double aV);// {return aV;}
};

template <> class TCompl<int>
{
    public :
        typedef double  TypeCompl;
        static int FromC(double aV) {return round_ni(aV);}
};
template <> class TCompl<double>
{
    public :
        typedef int  TypeCompl;
        static double FromC(int aV) {return aV;}
};
template <> class TCompl<float>
{
    public :
        typedef double  TypeCompl;
        static float FromC(double aV) {return (float)aV;}
};

template <> class TCompl<long double>
{
    public :
        typedef double  TypeCompl;
        static long double FromC(double aV) {return (long double)aV;}
};


template <class Type> class Pt2d : public  ElStdTypeScal<Type>
{
   public :

     typedef typename TCompl<Type>::TypeCompl  tCompl;
     typedef Pt2d<REAL>  TypeReel;
     typedef Type        TypeScal;
     typedef Pt2d<Type>  TypeEff;
     static Pt2d  El0 () {return Pt2d(0,0);}

     typename ElStdTypeScal<Type>::TypeScalReel Vol() const{return x*this->T2R(y);}

     typedef Pt2d<typename ElStdTypeScal<Type>::TypeVarProvReel> TypeProvPtScalR;

     typedef Type (& t2)[2] ;
     Type   x;
     Type   y;


  // Constructeur

     Pt2d<Type>()  : x (0), y (0) {}
     Pt2d<Type>(Type X,Type Y) : x (X), y (Y) {}

     Pt2d<Type>(const Pt2d<Type>& p) : x (p.x), y (p.y) {}
     explicit Pt2d<Type>(const Pt2d<tCompl>& p) :
              x( TCompl<Type>::FromC( p.x)),
              y( TCompl<Type>::FromC( p.y))
     {
     }

     static  Pt2d<Type> IP2ToThisT(const Pt2d<int> & aP){return Pt2d<Type>(Type(aP.x),Type(aP.y));}
     static  Pt2d<Type> RP2ToThisT(const Pt2d<double> & aP){return Pt2d<Type>(Type(aP.x),Type(aP.y));}
     static  Pt2d<Type> FP2ToThisT(const Pt2d<float> & aP){return Pt2d<Type>(Type(aP.x),Type(aP.y));}

/*

     Pt2d<Type>(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
     Pt2d<Type>(const Pt2d<REAL>& p): x (Pt2d<Type>::RtoT(p.x)), y (Pt2d<Type>::RtoT(p.y)) {};
*/


     static  Pt2d<Type>  FromPolar(REAL rho,REAL teta)
     {
        return   Pt2d<Type>(ElStdTypeScal<Type>::RtoT(cos(teta)*rho),ElStdTypeScal<Type>::RtoT(sin(teta)*rho));
     }

     static Pt2d<double> polar(const Pt2d<double> & p,REAL AngDef);

 // Operateurs

         // unaires,  Pt => Pt

     TypeProvPtScalR  ToPtProvR() const
     {
           return TypeProvPtScalR (this->T2R(x),this->T2R(y));
     }

     Pt2d<Type> operator - () const { return Pt2d<Type>(-x,-y); }
     Pt2d<Type> yx() const { return Pt2d(y,x);}
     Pt2d<Type> conj() const { return Pt2d(x,-y);}
     Pt2d<typename ElStdTypeScal<Type>::TypeScalReel> inv() const
     {
         typename ElStdTypeScal<Type>::TypeVarProvReel  n= this->T2R(x)*x+y*y;
         return Pt2d<typename ElStdTypeScal<Type>::TypeScalReel>(x/n,-y/n);
     };
      Pt2d<Type> Square() const;
      Type XtY() const {return x * y;}


         // binaires,  PtxPt => Pt

     Pt2d<Type> operator + (const Pt2d<Type> & p2) const
                {return Pt2d<Type>(x+p2.x,y+p2.y);}
     Pt2d<Type> operator * (const Pt2d<Type> & p2) const
                {return Pt2d<Type>(x*p2.x-y*p2.y,x*p2.y+y*p2.x);}

     // TCompl
     Pt2d<Type> operator / (const Pt2d<Type> & p2) const
     {
            TypeProvPtScalR aRes = this->ToPtProvR() * p2.inv().ToPtProvR();
             return Pt2d<Type> ((Type)aRes.x,(Type)aRes.y);
     }

     Pt2d<Type> operator - (const Pt2d<Type> & p2) const
                {return Pt2d<Type>(x-p2.x,y-p2.y);}
     Pt2d<Type> mcbyc(const Pt2d<Type> & p2) const
                {return Pt2d(x*p2.x,y*p2.y);}
     Pt2d<Type> dcbyc(const Pt2d<Type> & p2) const
                {return Pt2d(x/p2.x,y/p2.y);}


     void SetSup(const Pt2d<Type> & p){ElSetMax(x,p.x); ElSetMax(y,p.y);}
     void SetInf(const Pt2d<Type> & p){ElSetMin(x,p.x); ElSetMin(y,p.y);}

    // RatioMin :  return Min ( x/(TypeScalReel)p.x, y/(TypeScalReel)p.y);
     typename Pt2d<Type>::TypeScalReel RatioMin(const Pt2d<Type> & p) const;

         // binnaire, affectation composee

     Pt2d<Type> & operator += (const Pt2d<Type> & p2)
                { x += p2.x; y += p2.y; return * this;}
     Pt2d<Type> & operator -= (const Pt2d<Type> & p2)
                { x -= p2.x; y -= p2.y; return * this;}

     Pt2d<Type>  &  operator = (const Pt2d<Type> & p2)
     {
            x = p2.x;
            y = p2.y;
            return * this;
     }

         // binaire,  PtxPt => bool
     typename ElStdTypeScal<Type>::TypeBool  operator == (const Pt2d<Type> & p2) const {return (x==p2.x) && (y==p2.y);}
     typename ElStdTypeScal<Type>::TypeBool  operator != (const Pt2d<Type> & p2) const {return (x!=p2.x) || (y!=p2.y);}
     // p1 < p2 , utile par ex ds les map<Pt2di,Machin>
     typename ElStdTypeScal<Type>::TypeBool  operator <  (const Pt2d<Type> & p2) const {return (x<p2.x) || ((x==p2.x)&&(y<p2.y));}

     typename ElStdTypeScal<Type>::TypeBool   xety_inf_ou_egal (const Pt2d<Type> & p2) const
            {return (x<=p2.x) && (y<=p2.y);}

         // binaires,  PtxScalaire => Pt

     Pt2d<Type> operator * (INT  lambda) const { return Pt2d<Type>(x*lambda,y*lambda);}


     Pt2d<typename ElStdTypeScal<Type>::TypeScalReel> operator * (REAL lambda) const { return Pt2d<typename ElStdTypeScal<Type>::TypeScalReel>(x*lambda,y*lambda);}

     Pt2d<Type> operator / (INT  lambda) const { return Pt2d<Type>(x/lambda,y/lambda);}
     Pt2d<typename ElStdTypeScal<Type>::TypeScalReel> operator / (REAL lambda) const { return Pt2d<typename ElStdTypeScal<Type>::TypeScalReel>(x/lambda,y/lambda);}


      // operator * est deja surcharge
      Pt2d<Type> mul (const Type & aL) const { return Pt2d<Type>(x*aL,y*aL);}
      Pt2d<Type> div (const Type & aL) const { return Pt2d<Type>(x/aL,y/aL);}


         // binaires,  PtxPt => scalaire

     Type  operator ^ (const Pt2d<Type> & p2) const{return x*p2.y-y*p2.x;}


          // lies a une distance
    //friend Type  dist4(const Pt2d<Type> & p){return ElAbs(p.x)+  ElAbs(p.y);}
    //friend Type  dist8(const Pt2d<Type> & p){return ElMax(ElAbs(p.x),ElAbs(p.y));}

     typename ElStdTypeScal<Type>::TypeBool in_box(const Pt2d<Type> & p0, const Pt2d<Type> & p1)
     {
         return (x>=p0.x)&&(y>=p0.y)&&(x< p1.x)&&(y<p1.y);
     }

     friend void pt_set_min_max<>(Pt2d<Type> & p0,Pt2d<Type> & p1);

            // tertiaire

     // in_sect_angulaire :  est que le pt est dans le secteur partant de p1
     // et defini par un parcourt trigo jusqu'a p2
     bool in_sect_angulaire(const Pt2d<Type> & p1,const Pt2d<Type> & p2) const;

     // Ceux-ci n'ont aucun interet a etre iniline


     void to_tab(Type (& t)[2] ) const;
     static Pt2d<Type> FromTab(const Type *);
     static Pt2d<Type> FromTab(const std::vector<Type> &);
     std::vector<Type> ToTab() const;
     Output sigma();
     Output VMax();
     Output VMin();
     Output WhichMax();
     Output WhichMin();

     Pt2d<Type> AbsP() const {return Pt2d<Type>(ElAbs(x),ElAbs(y));}

     private :
          void Verif_adr_xy();

};

template <class Type>
Pt2d<double> Pt2d<Type>::polar( const Pt2d<double> & p,REAL AngDef )
{
    if ((p.x==0) && (p.y== 0))
        return Pt2d<double>(0,AngDef);
    return Pt2d<double>(hypot(p.x,p.y),atan2(p.y,p.x));
}

template <class Type>
 Type  dist4(const Pt2d<Type> & p){return ElAbs(p.x)+  ElAbs(p.y);}

template <class Type>
Type  dist8(const Pt2d<Type> & p){return ElMax(ElAbs(p.x),ElAbs(p.y));}

template <class Type>
Type  dist48(const Pt2d<Type> & p)
{
   Type Ax = ElAbs(p.x);
   Type Ay = ElAbs(p.y);
   return ElMax(Ax,Ay) + Ax + Ay;
}

template <class Type>
Type  dist48_euclid(const Pt2d<Type> & p)
{
   Type Ax = ElAbs(p.x);
   Type Ay = ElAbs(p.y);
   return (3*ElMax(Ax,Ay) +  2*(Ax + Ay)) / 5.0;
}





/*
template <> Pt2d<double>::Pt2d(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
template <> Pt2d<int>::Pt2d(const Pt2d<double>& p) : x (round_ni(p.x)), y (round_ni(p.y)) {};

template <> Pt2d<Fonc_Num>::Pt2d(const Pt2d<double>& p) : x (p.x), y (p.y) {};
*/
/*
     Pt2d<Type>(const Pt2d<INT>& p) : x (p.x), y (p.y) {};
     Pt2d<Type>(const Pt2d<REAL>& p): x (Pt2d<Type>::RtoT(p.x)), y (Pt2d<Type>::RtoT(p.y)) {};
*/

//Rotate aPt(X,Y) with angle(rad) and center(X,Y)
template <class Type>
Pt2d<Type> Rot2D(double aAngle, Pt2d<Type> aPt, Pt2d<Type> aRotCenter)
                 {  Pt2d<Type> PtOut;
                    PtOut.x=cos(aAngle)*(aPt.x-aRotCenter.x)+sin(aAngle)*(aPt.y-aRotCenter.y)+aRotCenter.x;
                    PtOut.y=-sin(aAngle)*(aPt.x-aRotCenter.x)+cos(aAngle)*(aPt.y-aRotCenter.y)+aRotCenter.y;
                    return PtOut;}


template <class Type>
void pt_set_min_max(Pt2d<Type> & p0,Pt2d<Type> & p1)
{
     set_min_max(p0.x,p1.x);
     set_min_max(p0.y,p1.y);
}

template <class Type>
Type scal(const Pt2d<Type> & p1,const Pt2d<Type> & p2)
{return p1.x*p2.x+p1.y*p2.y;}

template <class Type>
typename ElStdTypeScal<Type>::TypeScalReel  square_euclid(const Pt2d<Type> & p)
                 {return ElSquare(typename ElStdTypeScal<Type>::TypeScalReel(p.x)) + ElSquare(typename ElStdTypeScal<Type>::TypeScalReel(p.y));}

template <class Type>
typename ElStdTypeScal<Type>::TypeScalReel  square_euclid(const Pt2d<Type> & p1,const Pt2d<Type> & p2)
                 {return ElSquare(p2.x-p1.x) + ElSquare(p2.y-p1.y);}

template <class Type>
typename ElStdTypeScal<Type>::TypeScalReel  euclid(const Pt2d<Type> & p)
                 {return sqrt(square_euclid(p));}
template <class Type>
typename ElStdTypeScal<Type>::TypeScalReel  euclid(const Pt2d<Type> & p1,const Pt2d<Type> & p2)
                 {return sqrt(square_euclid(p1,p2));}

template <class Type> Pt2d<Type> Sup (const Pt2d<Type> & p1,const Pt2d<Type> & p2)
{
    return Pt2d<Type>(ElMax(p1.x,p2.x),ElMax(p1.y,p2.y));
}
template <class Type>Pt2d<Type> Inf(const Pt2d<Type> & p1,const Pt2d<Type> & p2)
{
    return Pt2d<Type>(ElMin(p1.x,p2.x),ElMin(p1.y,p2.y));
}

template <class Type> Pt2d<Type> Inf3 (const Pt2d<Type> & p1,const Pt2d<Type> & p2,const Pt2d<Type> & p3)
{
        return Inf(p1,Inf(p2,p3));
}
template <class Type>Pt2d<Type> Sup3 (const Pt2d<Type> & p1,const Pt2d<Type> & p2,const Pt2d<Type> & p3)
{
        return Sup(p1,Sup(p2,p3));
}


typedef  Pt2d<INT> Pt2di;
typedef  Pt2d<REAL> Pt2dr;
typedef  Pt2d<long double> Pt2dlr;
typedef  Pt2d<float> Pt2df;
typedef  Pt2d<U_INT2> Pt2dUi2;
double DMaxCoins(Pt2dr aSzIm,Pt2dr aC);
double DMaxCoins(Pt2dr aP0,Pt2dr aP1,Pt2dr aC);

template<class Type> Pt2dr ToPt2dr(const  Pt2d<Type> & aP) {return Pt2dr(aP.x,aP.y);}
template<class Type> Pt2df ToPt2df(const  Pt2d<Type> & aP) {return Pt2df(aP.x,aP.y);}
template<class Type> Pt2di ToPt2di(const  Pt2d<Type> & aP) {return Pt2di(round_ni(aP.x),round_ni(aP.y));}

extern std::ostream & operator << (std::ostream & ofs,const Pt2dUi2  &p);
extern std::ostream & operator << (std::ostream & ofs,const Pt2df  &p);
extern std::ostream & operator << (std::ostream & ofs,const Pt2dr  &p);
extern std::ostream & operator << (std::ostream & ofs,const Pt2di  &p);
extern std::istream & operator >> (std::istream & ifs,Pt2dr  &p);
extern std::istream & operator >> (std::istream & ifs,Pt2di  &p);


class cXml_Map2D;
class cXml_Map2DElem;
cXml_Map2D MapFromElem(const cXml_Map2DElem &);
class cXml_Homot;

class cElMap2D
{
    public :
         static cElMap2D * IdentFromType(int,const std::vector<std::string>* =0);
         virtual Pt2dr operator () (const Pt2dr & p) const = 0;
         virtual int Type() const = 0;
         virtual ~cElMap2D(){}
         virtual cElMap2D * Map2DInverse() const;
         virtual cElMap2D * Simplify() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cElMap2D * Duplicate() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cElMap2D * Identity() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier

         virtual int   NbUnknown() const;
         virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
         virtual void  InitFromParams(const std::vector<double> &aSol);


         void  SaveInFile(const std::string &);
         static cElMap2D * FromFile(const std::string &);
         virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0

          // Not yet commented
          void Affect(const cElMap2D &);
          virtual std::vector<double> Params() const;  // "Inverse" de InitFromParams
          virtual std::vector<std::string> ParamAux() const;  // Pour eventuellement param sec de Polyn
        private :
           virtual bool Compatible(const cElMap2D *) const; // Pour l'affectation, peut faire un down cast 
};

class cComposElMap2D : public cElMap2D
{
     public :
         virtual int Type() const ;
         cComposElMap2D(const std::vector<cElMap2D *>  & aVMap);


          static cComposElMap2D   NewFrom0();
          static cComposElMap2D   NewFrom1(cElMap2D *);
          static cComposElMap2D   NewFrom2(cElMap2D *,cElMap2D *);
          static cComposElMap2D   NewFrom3(cElMap2D *,cElMap2D *,cElMap2D*);


         virtual Pt2dr operator () (const Pt2dr & p) const ;
         virtual cElMap2D * Map2DInverse() const;
         virtual cElMap2D * Simplify() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
     public :
         std::vector<cElMap2D *> mVMap;
};


class ElHomot : public cElMap2D
{
      public :
         ElHomot(Pt2dr aTrans = Pt2dr(0,0), double aScale = 1.0) ;
         ElHomot(const cXml_Homot &) ;

         Pt2dr operator () (const Pt2dr & p) const
         {
               return  mTr + p * mSc;
         }
         ElHomot operator * (const ElHomot & sim2) const;

         virtual int Type() const ;
         virtual  cElMap2D * Map2DInverse() const;
         virtual cElMap2D * Duplicate() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cElMap2D * Identity() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
         ElHomot inv () const;

         virtual int   NbUnknown() const;
         virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
         virtual void  InitFromParams(const std::vector<double> &aSol);
         virtual std::vector<double> Params() const;  

         const Pt2dr  & Tr() const {return mTr;}
         const double & Sc() const {return mSc;}

      private :
        Pt2dr  mTr;
        double mSc;
};

class cXml_Homot;
ElHomot      Xml2EL(const cXml_Homot &);
cXml_Homot   EL2Xml(const ElHomot &);


class ElSimilitude : public cElMap2D
{
     public :

         ElSimilitude(Pt2dr trans = Pt2dr(0,0),Pt2dr ComplScale = Pt2dr(1,0)) :
            _tr (trans),
            _sc (ComplScale)
         {
         }

         static ElSimilitude SimOfCentre(Pt2dr centre,Pt2dr ComplScale)
         {
            return ElSimilitude
                   (
                        centre-centre*ComplScale,
                        ComplScale
                   );
         }

         static ElSimilitude SimOfCentre(Pt2dr centre,REAL rho,REAL teta)
         {
                return SimOfCentre
                       (
                           centre,
                           Pt2dr::FromPolar(rho,teta)
                       );
         }

         Pt2dr operator () (const Pt2dr & p) const
         {
               return _tr + p * _sc;
         }

         // sim1 * sim2 renvoie la similitude composee (celle z-> sim1(sim2(z)))

         ElSimilitude operator * (const ElSimilitude & sim2) const
         {
              return ElSimilitude
                     (
                         _tr + _sc * sim2._tr,
                         _sc * sim2._sc
                     );
         }

         virtual int   NbUnknown() const;
         virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
         virtual void  InitFromParams(const std::vector<double> &aSol);
         virtual std::vector<double> Params() const;  

         virtual int Type() const ;
         virtual  cElMap2D * Map2DInverse() const;
         virtual cElMap2D * Duplicate() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cElMap2D * Identity() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
         virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
         ElSimilitude inv () const
         {
              return ElSimilitude
                     (
                         (-_tr)/_sc,
                         _sc.inv()
                     );
         }

         Pt2dr tr () const {return _tr;}
         Pt2dr sc () const {return _sc;}

     private :
          Pt2dr  _tr;
          Pt2dr  _sc;
};

ElSimilitude  L2EstimSimHom(const class ElPackHomologue & aPack);



class cElHomographie;
class ElAffin2D : public cElMap2D
{
     public :
        ElAffin2D
        (
            Pt2dr im00,  // partie affine  -- translation
            Pt2dr im10,  // partie vecto
            Pt2dr im01  // partie vecto
        );


       static ElAffin2D  L2Fit(const class ElPackHomologue &,double * aRes=0);

        bool IsId() const;
        // bool operator == (const ElAffin2D & aF2);

        static ElAffin2D Id();
        static ElAffin2D trans(Pt2dr aTr);  // Ajoute Tr

  // Soit une image I1, que l'on Crop de Tr, puis que l'on sous echantillone
  // a d'une resolution aResol, pour avoir une image I2 renvoie la transfo qui donne les coordonnees
  // de l'homologue de I1 dans I2
  //
  //  Si aSzInOut est donne, on rajoute une eventuelle translation pour que l'image
  //  de la box est son coin en 0,0. La taille est modifiee et contient la taille finale
  //
  //
        static ElAffin2D TransfoImCropAndSousEch(Pt2dr aTr,Pt2dr aResol,Pt2dr * aSzInOut=0);  // Ajoute Tr
        static ElAffin2D TransfoImCropAndSousEch(Pt2dr aTr,double aResol,Pt2dr * aSzInOut=0);  // Ajoute Tr




        ElAffin2D (const ElSimilitude &);

        ElAffin2D();  // identite, on le laisse par compatibilite

        Pt2dr IVect (const Pt2dr & aP) const
        {
              return   mI10 *aP.x + mI01 *aP.y;
        }
        Pt2dr operator() (const Pt2dr & aP) const
        {
              return  mI00 + IVect(aP);
        }

         // idem sim Aff1 * Aff2 renvoie l'affinite e composee (celle z-> Aff1(Aff2(z)))
       ElAffin2D operator * (const ElAffin2D & sim2) const;
       ElAffin2D operator + (const ElAffin2D & sim2) const;
       ElAffin2D inv() const;

       virtual int   NbUnknown() const;
       virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
       virtual void  InitFromParams(const std::vector<double> &aSol);
       virtual std::vector<double> Params() const;  

       virtual  cElMap2D * Map2DInverse() const;
       virtual int Type() const ;
       virtual cElMap2D * Duplicate() ;  
       virtual cElMap2D * Identity() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
       virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0

       Pt2dr I00() const {return mI00;}
       Pt2dr I10() const {return mI10;}
       Pt2dr I01() const {return mI01;}
       static ElAffin2D FromTri2Tri
               (
                    const Pt2dr & a0, const Pt2dr & a1, const Pt2dr & a2,
                    const Pt2dr & b0, const Pt2dr & b1, const Pt2dr & b2
               );

       cElHomographie ToHomographie() const;

       // Ajoute une trans pout que aPt -> aRes
       ElAffin2D CorrectWithMatch(Pt2dr aPt,Pt2dr aRes) const;

     private :

            Pt2dr mI00;
            Pt2dr mI10;
            Pt2dr mI01;
};
double DMaxCoins(ElAffin2D AfC2M,Pt2dr aSzIm,Pt2dr aC);




// Fonctions specifiques a un des types de points

    // When a Pt2di p is used as a ``seed'' to generate a digital line the average euclidean
    // distance d between two consecutives points is variable according to p
    // For example  : d = sqrt(2) for p = (1,1) or p = (234,234) and d = 1 for p = (-99,0)
REAL  average_euclid_line_seed (Pt2di);
Pt2di  best_4_approx(const Pt2di & p);
Pt2di  second_freeman_approx(Pt2di u, bool conx_8,Pt2di u1);
INT    num_4_freeman(Pt2di);
Pt2dr ImAppSym(REAL A,REAL B,REAL C,Pt2dr aP);

Pt2di corner_box_included(Pt2di pmin,Pt2di pmax,bool left,bool down);

inline Pt2di round_ni(Pt2dr  p)
{
    return Pt2di(round_ni(p.x),round_ni(p.y));
}
inline Pt2di round_up(Pt2dr  p)
{
    return Pt2di(round_up(p.x),round_up(p.y));
}
inline Pt2di round_down(Pt2dr  p)
{
    return Pt2di(round_down(p.x),round_down(p.y));
}


inline Pt2di arrondi_sup(Pt2di a,Pt2di b)
{
    return Pt2di
           (
               arrondi_sup(a.x,b.x),
               arrondi_sup(a.y,b.y)
           );
}

inline Pt2di arrondi_sup(Pt2di a,int b)
{
   return arrondi_sup(a,Pt2di(b,b));
}



inline Pt2dr rot90(Pt2dr p)
{
    return Pt2dr(-p.y,p.x);
}
inline Pt2dr vunit(Pt2dr p,REAL & d)
{
   d = euclid(p);
   ELISE_ASSERT((d!=0),"Null seg in vunit");
   return p/d;
}

inline Pt2dr vunit(Pt2dr p)
{
   REAL d ;
   return vunit(p,d);
}

/*
inline Pt2dr barry(REAL pds1,const Pt2dr & p1,const Pt2dr & p2 )
{
     return p1*pds1  + p2*(1-pds1);
}
*/
template <class TPds,class TVal> inline TVal barry(TPds pds1,const TVal & p1,const TVal & p2 )
{
     return p1*pds1  + p2*(1-pds1);
}

#if (ELISE_ACTIVE_ASSER)
template <class Type> void assert_not_nul(const Pt2d<Type> & pt)
{
    ELISE_ASSERT((pt.x != 0) || (pt.y !=0),"Unexptected Nul point");
}
#else
template <class Type> void assert_not_nul(const Pt2d<Type> &){}
#endif


// angle avec vecteur (1,0), compris entre -pi et pi
// par ex (0,1) => pi / 2, si deux arg angle de p1 vers p2
// en fait juste encapsulation de atan2
REAL  angle(const Pt2dr & p);
REAL  angle(const Pt2dr & p1,const Pt2dr & p2);


// La fonction polar est maintenant dans la classe Pt2d !!!
// Pt2dr polar(const Pt2dr & p,REAL AngDef);

// angle de droite (entre -pi/2 et pi/2),
//  angle de droite non oriente (entre 0 et pi/2, symetrique)

REAL  angle_de_droite(const Pt2dr & p);
REAL  angle_de_droite(const Pt2dr & p1,const Pt2dr & p2);
REAL  angle_de_droite_nor(const Pt2dr & p);
REAL  angle_de_droite_nor(const Pt2dr & p1,const Pt2dr & p2);



// Fonction assez lentes , a utiliser pour memoriser
std::vector<Pt2di> PointInCouronne(int aD8Min,int aD8Max);
    // Par ex dist [2,5,9] et recuper (0-1) + (2-4) + (5-8)
    // AddD4First mets les 4 voisin d'abord
std::vector<std::vector<Pt2di> > PointOfCouronnes(const std::vector<int> &Dist,bool AddD4First);

std::vector<std::vector<Pt2di> > StdPointOfCouronnes(int aDMax,bool AddD4First);


#define Pt3di  Pt3d<INT>
#define Pt3dr  Pt3d<REAL>
#define Pt3df  Pt3d<float>


template <class Type> class Pt3d : public  ElStdTypeScal<Type>
{
   public :
     typedef typename TCompl<Type>::TypeCompl  tCompl;
     Type   x;
     Type   y;
     Type   z;

     Pt3d();

     Pt3d<Type>(const Pt3d<Type>& ); // to please visual
     explicit Pt3d<Type>(const Pt3d<tCompl>& p);

     static  Pt3d<Type> P3ToThisT(const Pt3d<int> & aP){return Pt3d<Type>(Type(aP.x),Type(aP.y),Type(aP.z));}
     static  Pt3d<Type> P3ToThisT(const Pt3d<double> & aP){return Pt3d<Type>(Type(aP.x),Type(aP.y),Type(aP.z));}
     static  Pt3d<Type> P3ToThisT(const Pt3d<float> & aP){return Pt3d<Type>(Type(aP.x),Type(aP.y),Type(aP.z));}


     Pt3d<Type>(const Pt2d<Type>&,Type z); // to please visual

     Pt3d<Type> mcbyc(const Pt3d<Type> & p2) const
                {return Pt3d(x*p2.x,y*p2.y,z*p2.z);}
     Pt3d<Type> dcbyc(const Pt3d<Type> & p2) const
                {return Pt3d(x/p2.x,y/p2.y,z/p2.z);}

     static Pt3d<Type> RandC() {return Pt3d<Type>(NRrandC(),NRrandC(),NRrandC());}
     static Pt3d<Type> Rand3() {return Pt3d<Type>(NRrandom3(),NRrandom3(),NRrandom3());}

     Pt3d(Type X,Type Y,Type Z);
     Pt3d<Type> operator + (const Pt3d<Type> & p2) const;

     Pt3d<Type> operator * (Type) const;
     Pt3d<Type> operator / (Type) const;

     Pt3d<Type> operator - (const Pt3d & p2) const;
     Pt3d<Type> operator - () const;
     typename ElStdTypeScal<Type>::TypeScalReel Vol() const{return x*(y*this->T2R(z));}
     Pt3d<Type> PVolTarget(double aVolTarget) const {return (*this) * pow(aVolTarget/Vol(),1/3.0);}
     Pt3d<Type> PVolUnite() const {return PVolTarget(1.0);}

     typename ElStdTypeScal<Type>::TypeBool  operator == (const Pt3d<Type> & p2) const {return (x==p2.x) && (y==p2.y) && (z==p2.z);}

     // multiplication coordinate by coordinate

     // friend Type  scal<Type> (const Pt3d<Type> & p1,const Pt3d<Type> & p2);
/*
     friend Type Det(const Pt3d<Type> & p1,const Pt3d<Type> & p2,const Pt3d<Type> & p3)
     {
         return scal(p1 ,p2^p3);
     }
*/

     Pt3d<Type>  operator ^ (const Pt3d<Type> & p2) const;
     Pt3d<Type>  &  operator = (const Pt3d<Type> & p2) ;


     void to_tab(Type (& t)[3] ) const;
     static Pt3d<Type> FromTab(const Type *);
     std::vector<Type> ToTab() const;
     static Pt3d<Type> FromTab(const std::vector<Type> &);

     Pt3d<Type> AbsP() const {return Pt3d<Type>(ElAbs(x),ElAbs(y),ElAbs(z));}
     /*
     friend Pt3d<Type> Sup (const Pt3d<Type> & p1,const Pt3d<Type> & p2)
           { return Pt3d<Type>(ElMax(p1.x,p2.x),ElMax(p1.y,p2.y),ElMax(p1.z,p2.z));}
     friend Pt3d<Type> Inf (const Pt3d<Type> & p1,const Pt3d<Type> & p2)
            { return Pt3d<Type>(ElMin(p1.x,p2.x),ElMin(p1.y,p2.y),ElMin(p1.z,p2.z));}
      */
     Output sigma();
     Output VMax();
     Output VMin();
     Output WhichMax();
     Output WhichMin();

     static Type instantiate();



     // ! Convention Phi = 0 "a l'equateur"
     static  Pt3d<Type>  TyFromSpherique(Type Rho,Type Teta,Type Phi)
     {
        return   Pt3d<Type>
         (
          ElStdTypeScal<Type>::RTtoT(cos(Phi)*cos(Teta)*Rho),
          ElStdTypeScal<Type>::RTtoT(cos(Phi)*sin(Teta)*Rho),
          ElStdTypeScal<Type>::RTtoT(sin(Phi)*Rho)
         );
     }

     private :
          void Verif_adr_xy();
};

template <class Type>
Pt3d<Type> Sup (const Pt3d<Type> & p1,const Pt3d<Type> & p2)
{ return Pt3d<Type>(ElMax(p1.x,p2.x),ElMax(p1.y,p2.y),ElMax(p1.z,p2.z));}
template <class Type>
Pt3d<Type> Inf (const Pt3d<Type> & p1,const Pt3d<Type> & p2)
{ return Pt3d<Type>(ElMin(p1.x,p2.x),ElMin(p1.y,p2.y),ElMin(p1.z,p2.z));}

inline Pt3dr Pcoord2(const Pt3dr & aP) { return Pt3dr(ElSquare(aP.x),ElSquare(aP.y),ElSquare(aP.z)); }
inline Pt2dr Pcoord2(const Pt2dr & aP) { return Pt2dr(ElSquare(aP.x),ElSquare(aP.y)); }

inline double SomCoord(const Pt3dr & aP) { return aP.x+aP.y+aP.z;}
inline double SomCoord(const Pt2dr & aP) { return aP.x+aP.y;}


bool BadValue(const Pt3dr &);
bool BadValue(const Pt2dr &);
bool BadValue(const double &);


template <class Type> typename ElStdTypeScal<Type>::TypeScalReel  square_euclid(const Pt3d<Type> & p)
     {
        typename ElStdTypeScal<Type>::TypeScalReel aX = p.x;
        typename ElStdTypeScal<Type>::TypeScalReel aY = p.y;
        typename ElStdTypeScal<Type>::TypeScalReel aZ = p.z;

        return aX*aX + aY*aY + aZ*aZ;
     }
template <class Type>  typename ElStdTypeScal<Type>::TypeScalReel  euclid(const Pt3d<Type> & p ){return sqrt(square_euclid(p));}
template <class Type> typename ElStdTypeScal<Type>::TypeScalReel  square_euclid(const Pt3d<Type> & p1,const Pt3d<Type> & p2)
{
    return ElSquare(p2.x-p1.x) + ElSquare(p2.y-p1.y) + ElSquare(p2.z-p1.z);
}
template <class Type>
Type  scal (const Pt3d<Type> & p1,const Pt3d<Type> & p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

template <class Type>
Type Det(const Pt3d<Type> & p1,const Pt3d<Type> & p2,const Pt3d<Type> & p3)
{
    return scal(p1 ,p2^p3);
}


Pt3dr OneDirOrtho(const Pt3dr &);  // Vecteur unitaire

template <class Type> inline Pt2d<Type> Proj(Pt3d<Type> aP) {return Pt2d<Type>(aP.x,aP.y);}

template <class Type> Pt2d<Type> ProjStenope(Pt3d<Type> aP)
{return Pt2d<Type>(aP.x,aP.y)/aP.z;}


template <class Type> Pt3d<Type> PtAndZ(Pt2d<Type> aP,Type aZ)
{return Pt3d<Type>(aP.x,aP.y,aZ);}

template <class Type> Pt3d<Type> PZ1(Pt2d<Type> aP)
{return Pt3d<Type>(aP.x,aP.y,(Type)1);}
template <class Type> Pt3d<Type> PZ0(Pt2d<Type> aP)
{return Pt3d<Type>(aP.x,aP.y,(Type)0);}

template <class Type> Pt3d<Type> PointNorm1(const Pt3d<Type> & aP)
{return aP / sqrt(aP.x*aP.x+aP.y*aP.y + aP.z*aP.z);}

inline Pt3di round_down(Pt3dr  p)
{
    return Pt3di(round_down(p.x),round_down(p.y),round_down(p.z));
}


Pt3dr vunit(const Pt3dr & p);

// return rho
REAL ToSpherique(const Pt3dr & aP,REAL & rho,REAL & teta,REAL & phi);

template <class Type> void  corner_boxes
                            (
                                          Pt3d<Type> p1,
                                          Pt3d<Type> p2,
                                          Pt3d<Type> (& t)[8]
                            );
template <class Type>  void pt_set_min_max(Pt3d<Type> & p0,Pt3d<Type> & p1);


extern std::ostream & operator << (std::ostream & ofs,const Pt3dr  &p);
extern std::ostream & operator << (std::ostream & ofs,const Pt3di  &p);
extern std::istream & operator >> (std::istream & ifs,Pt3dr  &p);
extern std::istream & operator >> (std::istream & ifs,Pt3di  &p);

#define Pt4di  Pt4d<INT>
#define Pt4dr  Pt4d<REAL>


template <class Type> class Pt4d
{
   public :
     Type   x;
     Type   y;
     Type   z;
     Type   t;

     Pt4d();
     Pt4d(Type X,Type Y,Type Z,Type T);

     static Type instantiate();

     private :
//          void Verif_adr_xy();
// Methode jamais ecrite
};





class ElCmpZ
{
   public :
    bool operator()(const Pt3di& p1, const Pt3di& p2) const { return p1.z < p2.z; }
};

class ElCmp4Z
{
   public :
    bool operator()(const Pt4di& p1, const Pt4di& p2) const { return p1.z < p2.z; }
};

/* JYCAR
class ElInd4Z
{
   public :
    bool operator()(const Pt4di& p1) const { return p1.z; }
};
*/


class Interval
{
   public :

     REAL _v0;   // begin
     REAL _v1;   // end

     Interval(REAL v0,REAL v1);
     Interval();
     REAL dist(const Interval &);
};


template <class Type> class BoxFreemanCompil;

template <class Type> Box2d<Type> Sup(const Box2d<Type> & b1, const Box2d<Type> & b2);
template <class Type> Box2d<Type> Inf(const Box2d<Type> & b1, const Box2d<Type> & b2);
template <class Type> bool InterVide(const Box2d<Type> & b1, const Box2d<Type> & b2);


inline Pt2dr ToPt2dr(const Pt2dr & aP) {return aP;}
inline Pt2dr ToPt2dr(const Pt2di & aP) {return Pt2dr(aP.x,aP.y);}
inline Pt2dr ToPt2dr(const Pt2dlr & aP){return Pt2dr(aP.x,aP.y);}
inline Pt2di ToPt2di(const Pt2dr & aP) {return Pt2di(round_ni(aP.x),round_ni(aP.y));}
inline Pt2di ToPt2di(const Pt2di & aP) {return aP;}
inline Pt2di ToPt2di(const Pt2dlr & aP){return Pt2di(round_ni(aP.x),round_ni(aP.y));}


class Flux_Pts;
template <class Type> class Box2d
{
   public :

      Box2d<double> BoxImage(const cElMap2D &) const;

      Interval  XInterv() const {return Interval(_p0.x,_p1.x);}
      Interval  YInterv() const {return Interval(_p0.y,_p1.y);}

     typedef  Box2d<Type> QBox[4];
     typedef  Pt2d<Type>  P4[4];

     Pt2dr  RandomlyGenereInside() const;

     std::vector<Pt2dr> ClipConpMax(const std::vector<Pt2dr> &);


     Pt2d<Type>  _p0;
     Pt2d<Type>  _p1;
     Pt2d<Type> milieu() const { return (_p0+_p1) / 2;}
     Pt2d<Type> sz() const { return _p1 - _p0;}
     Pt2d<Type> FromCoordLoc(Pt2dr aP) const { return Pt2d<Type>(ToPt2dr(_p0)+aP.mcbyc(ToPt2dr(sz())));}


     Pt2dr ToCoordLoc(Pt2dr aP) const { return (aP-ToPt2dr(_p0)).dcbyc(ToPt2dr(sz()));}

     Type   hauteur() const { return _p1.y-_p0.y;}
     Type   largeur() const { return _p1.x-_p0.x;}
     REAL  diam() const{return euclid(_p0,_p1);}
     Type surf() {return hauteur() * largeur();}
     Type x(int i) const {return i ? _p1.x : _p0.x;}
     Type y(int i) const {return i ? _p1.y : _p0.y;}

     Box2d<Type> trans(Pt2d<Type>) const;
     Pt2d<Type> P0() const {return _p0;}
     Pt2d<Type> P1() const {return _p1;}
     Flux_Pts Flux() const;


     // 0 a l'exterieur, distance (d8) au bord a l'interieur
     double Interiorite(const Pt2dr & aP) const;


     Box2d(){}
     Box2d(Type);
     Box2d(Pt2d<Type>);
     Box2d(const Pt2d<Type> *,INT aNb);
     Box2d(Pt2di,Pt2di);
     Box2d(Pt2dr,Pt2dr);  // cast up and down
     Box2d(Pt2dlr,Pt2dlr);  // cast up and down
     Box2d(const Type *,const Type *,INT);
     bool include_in(const Box2d<Type> & b2) const;
     Box2d<Type>  erode(Pt2d<Type>) const;
     Box2d<Type>  dilate(Pt2d<Type>) const;
     Box2d<Type>  dilate(Type) const;

     std::vector<Pt2d<Type> >   Contour() const;


     // + ou - dilatation signee, en fait equivalent avec la
     // definition actuelle de dilate (mais le cote algebrique de
     // de dilate n'est pas acquis a 100%)
     Box2d<Type>  AddTol(const Box2d<Type> &) const;
     Box2d<Type>  AddTol(const Pt2d<Type> &) const;
     Box2d<Type>  AddTol(const Type &) const;

     bool  inside(const Pt2d<Type> & p) const;  // p0 <= Box._p1
     bool  inside_std(const Pt2d<Type> & p) const;  // p0 < Box._p1

     bool contains(const Pt2d<int> & pt) const
     {
        return (pt.x>=_p0.x) && (pt.y>=_p0.y) && (pt.x<_p1.x) && (pt.y<_p1.y);
     }
     bool contains(const Pt2d<double> & pt) const
     {
        return (pt.x>=_p0.x) && (pt.y>=_p0.y) && (pt.x<_p1.x) && (pt.y<_p1.y);
     }
     bool contains(const Pt2d<long double> & pt) const
     {
        return (pt.x>=_p0.x) && (pt.y>=_p0.y) && (pt.x<_p1.x) && (pt.y<_p1.y);
     }



     Pt2dr FromCoordBar(Pt2dr aCBar) const;


   //   QT

          // generaux

          INT  freeman_pos(const Pt2dr & pt) const;

          // box point

     bool   Intersecte(const Pt2dr &) const;
     bool   Intersecte(const SegComp &) const;
     bool   Intersecte(const Seg2d &) const;
     bool   Intersecte(const cElTriangleComp &) const;
     bool   Include(const Pt2dr &) const;
     bool   Include(const SegComp &) const;
     bool   Include(const Seg2d &) const;
     bool   Include(const cElTriangleComp &) const;
     REAL8  SquareDist(const Pt2dr &) const;
     REAL8  SquareDist(const SegComp &) const;
     REAL8  SquareDist(const Seg2d &) const;
     REAL8  SquareDist(const Box2d<Type> &) const;

     void   QSplit(QBox &) const; // Split in 4 box (for Q-Tree)
     void   QSplitWithRab(QBox &,Type aRab) const; // Split in 4 box (for Q-Tree)
     void   Corners(P4 &) const; // Split in 4 box (for Q-Tree)

      void PtsDisc(std::vector<Pt2dr> &,INT aNbPts);
   private :
        typedef REAL8 (Box2d<Type>:: * R_fonc_Pt2dr)(const Pt2dr &) const;

        REAL  Freem0SquareDist(const Pt2dr &) const;
        REAL  Freem1SquareDist(const Pt2dr &) const;
        REAL  Freem2SquareDist(const Pt2dr &) const;
        REAL  Freem3SquareDist(const Pt2dr &) const;
        REAL  Freem4SquareDist(const Pt2dr &) const;
        REAL  Freem5SquareDist(const Pt2dr &) const;
        REAL  Freem6SquareDist(const Pt2dr &) const;
        REAL  Freem7SquareDist(const Pt2dr &) const;
        REAL  Freem8SquareDist(const Pt2dr &) const;

        friend  class BoxFreemanCompil<Type>;
        REAL8  SquareDist(const Pt2dr &,INT c) const;

        static R_fonc_Pt2dr _Tab_FreemSquareDist[9];
};



typedef Box2d<INT> Box2di;
typedef Box2d<REAL>  Box2dr;
cElMap2D *  MapPolFromHom(const ElPackHomologue & aPack,const Box2dr & aBox,int aDeg,int aRabDegInv);
Pt2di BoxPClipedIntervC(const Box2di &,const Pt2di &);

extern std::istream & operator >> (std::istream & ifs,Box2dr  &aBox);
extern std::istream & operator >> (std::istream & ifs,Box2di  &aBox);
Pt2di  RandomlyGenereInside(const Box2di &) ;

Box2dr  I2R(const Box2di &);
Box2di  R2I(const Box2dr &);   // Par round_ni
Box2di  R2ISup(const Box2dr &);   // Par down et sup

ostream & operator << (ostream & ofs,const Box2di  &aBox);
ostream & operator << (ostream & ofs,const Box2dr  &aBox);



void AdaptParamCopyTrans(INT& X0src,INT& X0dest,INT& NB,INT NbSrc,INT NbDest);

void AdaptParamCopyTrans(Pt2di& p0src,Pt2di& p0dest,Pt2di& sz,
                          Pt2di   SzSrc, Pt2di   SzDest);


template <class Type>
Box2d<Type> I2Box
       (
        const cInterv1D<Type>  & IntervX,
        const cInterv1D<Type>  & IntervY
       );
template <class Type> cInterv1D<Type> IntX(const Box2d<Type> &);
template <class Type> cInterv1D<Type> IntY(const Box2d<Type> &);

class cDecoupageInterv2D
{
      public :

          // dil peut etre < 0, dilate sauf
          Box2di DilateBox(int aKBox,const Box2di &,int aDil);


          cDecoupageInterv2D
          (
              const Box2di & aBoxGlob,
              Pt2di aSzMax,
              const Box2di   & aSzBord,
              int              anArrondi=1
          );
          static cDecoupageInterv2D SimpleDec(Pt2di aSz,int aSzMax,int aSzBrd,int anArrondi=1);

          int NbInterv() const;
          Box2di KthIntervOut(int aK) const;
          Pt2di  IndexOfKBox(int aKBOx) const;

      // Avec Bord par defaut
          Box2di  KthIntervIn(int aK) const;
          Box2di  KthIntervIn(int aK, const Box2di   & aSzBord) const;
       // Majorant de la taille des boites
      Pt2di   SzMaxOut() const;
      Pt2di   SzMaxIn (const Box2di   & aSzBord) const;
      Pt2di   SzMaxIn () const;
      int     NbX() const;
      private :
          cDecoupageInterv1D mDecX;
          cDecoupageInterv1D mDecY;
      int mNbX;
      Box2di mSzBrd;

};


extern const Pt2dr aPRefFullFrame;
class cElRegex;
class cMetaDataPhoto
{
    public :

        bool  IsNoMTD() const;


        // Valeur par laquelle il faut mulpitlier elle meme pour egaliser Ref
        double MultiplierEqual(const cMetaDataPhoto &,bool * AllOk) const;

        // static const cMetaDataPhoto &  CreateExiv2(const std::string &,const char * aNameTest=0);
        static const cMetaDataPhoto &  CreateExiv2(const std::string &);
        const cElDate & Date(bool Svp=false) const;
        void SetSz(const Pt2di &);
        void SetFocal(const double &);
        void SetFoc35(const double &);
        void SetCam(const std::string &);
        double FocMm(bool Svp=false) const;
        double Foc35(bool Svp=false) const;
        double  FocPix() const;
        int NbBits(bool Svp=false) const;

        double ExpTime(bool Svp=false) const;
        double Diaph(bool Svp=false) const;
        double IsoSpeed(bool Svp=false) const;
        const std::string &  Cam(bool Svp=false) const;

        Pt2di XifSzIm(bool Svp=false) const;
        Pt2di TifSzIm(bool Svp=false) const;
        Pt2di SzImTifOrXif(bool Svp=false) const;

        void SetXYZTetas(const Pt3dr & aXYZ,const Pt3dr & Tetas);
        bool XYZTetasInit() const;
        const Pt3dr & XYZ() const;
        const Pt3dr & Tetas() const;


         const bool   &  HasGPSLatLon() const;
         const double &  GPSLat() const;
         const double &  GPSLon() const;
         const bool   &  HasGPSAlt() const;
         const double &  GPSAlt() const;
         void SetGPSLatLon(const double & aLat,const double & aLon);
         void SetGPSAlt(const double & anAlt);

         cMetaDataPhoto
         (
                const std::string & aNameIm,
                Pt2di aSzIm,
                const std::string & aCam,
                cElDate mDate,double aFocMm,double Foc35,double aExpTime,
                double aDiaph,double anIsoSpeed,const std::string & aBayPat,
                const std::string & anOrientation, const std::string & aCameraOrientation,
                int aNbBits
         );
         cMetaDataPhoto();
         const std::string  & BayPat() const;
         bool & FocForced();
         const std::string & Orientation() const;
         const std::string & CameraOrientation() const;
         void dump( const std::string &aPrefix, std::ostream &aStream = std::cout );
   private :
        static cMetaDataPhoto  CreateNewExiv2(const std::string &);

         static cMetaDataPhoto Create(const std::string & aCom,const std::string &);
         static std::string  ExeCom(const std::string & aNameProg,const std::string & aNameFile);

         static cElRegex * mDateRegExiV2;
         static cElRegex * mFocRegExiV2;
         static cElRegex * mFoc35RegExiV2;
         static cElRegex * mExpTimeRegExiV2;
         static cElRegex * mDiaphRegExiV2;
         static cElRegex * mIsoSpeedRegExiV2;
         static cElRegex * mCameraExiV2;
         static cElRegex * mSzImExiV2;
         static const std::string theNameTMP;

         static const cMetaDataPhoto  TheNoMTD;

         std::string  mNameIm;
         mutable Pt2di   mTifSzIm;
         Pt2di   mXifSzIm;
         std::string mCam;
         cElDate mDate;
         double  mFocMm;
         bool    mFocForced;
         double  mFoc35;
         double  mExpTime;
         double  mDiaph;
         double  mIsoSpeed;
         bool    mXYZ_Init;
         bool    mTeta_Init;
         Pt3dr   mXYZ;
         Pt3dr   mTetas;
         std::string  mBayPat;
         bool    mHasGPSLatLon;
         double  mGPSLat;
         double  mGPSLon;
         bool    mHasGPSAlt;
         double  mGPSAlt;
         std::string mOrientation;
         std::string mCameraOrientation;
         int         mNbBits;  // Par defaut initialisee a -1

};
// cCameraEntry *  CamOfName(const std::string & aName);

extern double GetFocalMmDefined(const std::string & aNameFile);

extern bool CmpY(const Pt2di & aP1,const Pt2di & aP2);



class cSystemeCoord;
class cChangementCoordonnees;
class cBasicSystemeCoord;
class cXmlGeoRefFile;


class cSysCoordPolyn;
template <class Type>  class ElMatrix;


class cSysCoord
{
     public :

         ElMatrix<double> JacobToGeoc(const Pt3dr &,const Pt3dr& Epsilon = Pt3dr(0.1,0.1,0.1) ) const;
         ElMatrix<double> JacobFromGeoc(const Pt3dr &,const Pt3dr& Epsilon = Pt3dr(0.1,0.1,0.1)) const;



         // Au moins un des deux ToGeoC doit etre defini, car les version par defaut definissent l'un par rapport
         // a l'autre, d'ou possible recursion infinie ....
         virtual Pt3dr ToGeoC(const Pt3dr &) const ;
         virtual std::vector<Pt3dr> ToGeoC(const std::vector<Pt3dr> &) const ;

         // Idem from GeoC
         virtual Pt3dr FromGeoC(const Pt3dr &) const ;
         virtual std::vector<Pt3dr> FromGeoC(const std::vector<Pt3dr> &) const ;






         Pt3dr FromSys2This(const cSysCoord &,const Pt3dr &) const;

         ElMatrix<double> JacobSys2This(const cSysCoord &,const Pt3dr &,const Pt3dr& Epsilon = Pt3dr(0.1,0.1,0.1)) const;

         virtual cSystemeCoord ToXML() const = 0;

          virtual Pt3dr OdgEnMetre() const = 0;  // Ordre dde grandeir en metre
                                                 //  tq. p.x est la valeur donnant en ordre de grandeur un dep de 1

          static cSysCoord * GeoC();
          static cSysCoord * WGS84();
          static cSysCoord * WGS84Degre();
          static cSysCoord * RTL(const Pt3dr & Ori);
          static cSysCoord * FromXML(const cSystemeCoord &,const char * aDir);

          static cSysCoord * FromFile(const std::string & aNF,const std::string & aTag="SystemeCoord");

          static cSysCoord * ModelePolyNomial
                             (
                                    Pt3di aDegX,
                                    Pt3di aDegY,
                                    Pt3di aDegZ,
                                    cSysCoord * aSysIn,
                                    const std::vector<Pt3dr> & aVin,
                                    const std::vector<Pt3dr> & aVout
                             );

          static cSysCoordPolyn * TypedModelePolyNomial
                             (
                                    Pt3di aDegX,
                                    Pt3di aDegY,
                                    Pt3di aDegZ,
                                    cSysCoord * aSysIn,
                                    const std::vector<Pt3dr> & aVin,
                                    const std::vector<Pt3dr> & aVout
                             );

          virtual void Delete() = 0;  //  Virtuel car certain sont "indestructible"

         ElMatrix<double> Jacobien(const Pt3dr &,const Pt3dr& Epsilon,bool SensToGeoC) const;
         std::vector<ElMatrix<double> > Jacobien(const std::vector<Pt3dr > &,const Pt3dr& Epsilon,bool SensToGeoC,std::vector<Pt3dr> * aVPts=0 ) const;

     protected :
           virtual ~cSysCoord();
     private :

         Pt3dr  Transfo(const Pt3dr &, bool SensToGeoC) const;
         std::vector<Pt3dr>  Transfo(const std::vector<Pt3dr> & aV, bool SensToGeoC) const;



          static cSysCoord * FromXML
                             (
                                   const cBasicSystemeCoord * &,
                                   int  & aNbB,
                                   const char * aDir
                             );
};

class cProj4 : public cSysCoord
{
    public :
        cProj4(const std::string  & aStr,const Pt3dr & aMOdg);

        std::vector<Pt3dr> ToGeoC(const std::vector<Pt3dr> &) const;
        std::vector<Pt3dr> FromGeoC(const std::vector<Pt3dr> &) const ;

        static cProj4  Lambert(double aPhi0,double aPhi1,double aPhi2,double aLon0,double aX0,double aY0);
        static cProj4 * Lambert93();

        Pt3dr OdgEnMetre() const;
        cSystemeCoord ToXML() const;
        
        void Delete();

    private :
        std::vector<Pt3dr> Chang(const std::vector<Pt3dr> &, bool Sens2GeoC) const;
        std::string mStr;
        Pt3dr       mMOdg;

};

class cCs2Cs //:  public cSysCoord
{
    public :
        cCs2Cs(const std::string  & aStr);

        std::vector<Pt3dr> Chang(const std::vector<Pt3dr> &) const;

        cSystemeCoord ToXML() const;

        void Delete();

    private :
        std::string mStr;

};

class ElCamera;

class cTransfo3D
{
     public :
          virtual std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> &) const = 0;
          static cTransfo3D * Alloc(const std::string & aName,const std::string & aDir) ;

};

class cChSysCo : public cTransfo3D
{
     public :
           Pt3dr Src2Cibl(const Pt3dr &) const;
           Pt3dr Cibl2Src(const Pt3dr &) const;
           std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> &) const;
           std::vector<Pt3dr> Cibl2Src(const std::vector<Pt3dr> &) const;
           static cChSysCo * Alloc(const std::string & aName,const std::string & aDir) ;

           void ChangCoordCamera(const std::vector<ElCamera *> & aVCam,bool ForceRot);
           //   cChSysCo(const cChangementCoordonnees &,const std::string &) ;
           cChSysCo(cSysCoord * aSrc,cSysCoord * aCibl);
     private :
           ~cChSysCo();
           cSysCoord * mSrc;
           cSysCoord * mCibl;
};


class cGeoRefRasterFile
{
     public :
        cGeoRefRasterFile(const cXmlGeoRefFile &,const char * aDir);
        static cGeoRefRasterFile * FromFile(const std::string & aNF,const std::string & aTag="XmlGeoRefFile");

        Pt3dr File2Loc(const Pt3dr & ) const;
        Pt3dr File2Loc(const Pt2dr & ) const;  // Valide si ZMoyen
        Pt3dr File2GeoC(const Pt3dr & ) const;
        Pt3dr File2GeoC(const Pt2dr & ) const;


        Pt3dr Loc2File(const Pt3dr & ) const;
        Pt3dr Geoc2File(const Pt3dr & ) const;

        double ZMoyen() const; // N'existe pas toujours
     private :
        Pt3dr Raster2DTo3D(const Pt2dr & aP) const;
        void AssertSysCo() const;
        void AssertZMoy() const;

        cSysCoord * mSys;
        bool        mHasZMoy;
        Pt2dr       mOriXY;
        Pt2dr       mResolXY;
        double      mOriZ;
        double      mResolZ;
};

Pt3dr  tCho2double(const Pt3d<tSysCho> & aP);


typedef TypeSubst<Pt2di>   Pt2diSubst;
typedef TypeSubst<Pt2dr>   Pt2drSubst;

std::vector<Pt3dr>  GetDistribRepreBySort(std::vector<Pt2dr> & aVP,const Pt2di & aNbOut,Pt3dr & aPRep);

std::vector<Pt3dr> GetDistribRepresentative(Pt3dr & aCdg,const std::vector<Pt2dr> & aV,const Pt2di & aNb);


namespace std
{
bool operator < (const Pt3di & aP1,const Pt3di & aP2);
}


class cMTDImCalc;
class cMIC_IndicAutoCorrel;

cMTDImCalc GetMTDImCalc(const std::string & aNameIm);
const cMIC_IndicAutoCorrel * GetIndicAutoCorrel(const cMTDImCalc & aMTD,int aSzW);
std::string NameMTDImCalc(const std::string & aFullName,bool Bin);


inline double CoutAttenueTetaMax(const double & aVal,const double & aVMax)
{
      return  (aVal*aVMax) / (aVal + aVMax);
}

inline double GenCoutAttenueTetaMax(const double & aVal,const double & aVMax)
{
      if (aVMax<=0) return aVal;
      return CoutAttenueTetaMax(aVal,aVMax);
}

Pt2dr arrondi_ni(const Pt2dr & aP,double aPer);

inline std::string to_yes_no( bool aBoolean ){ return aBoolean ? "yes" : "no"; }

inline std::string to_true_false( bool aBoolean ){ return aBoolean ? "true" : "false"; }

template <class Type,class TypePt> inline int  CmpValAndDec(const Type & aV1,const Type & aV2, const TypePt & aDec)
{
   //    aV1 =>   aV1 + eps * aDec.x + eps * esp * aDec

   if (aV1 < aV2) return -1;
   if (aV1 > aV2) return  1;

   if (aDec.x<0)  return -1;
   if (aDec.x>0)  return  1;

   if (aDec.y<0)  return -1;
   if (aDec.y>0)  return  1;

   return 0;
}

std::vector<Pt2di> SortedVoisinDisk(double aDistMin,double aDistMax,bool Sort);

template <class Type> int DimPts(Pt2d<Type> *) {return 2;}
template <class Type> int DimPts(Pt3d<Type> *) {return 3;}
template <class Type> int DimPts(Pt4d<Type> *) {return 4;}


template <class Type> class cCmpPtOnAngle
{
    public :
         bool  operator()(const Type & aP1 ,const Type & aP2)
         {    
              Pt2dr aPol1 = Pt2dr::polar(Pt2dr(aP1),1);
              Pt2dr aPol2 = Pt2dr::polar(Pt2dr(aP2),1);
              return aPol1.y < aPol2.y;
         }
};

class cSegEntierHor
{
    public :
        Pt2di mP0;
        int   mNb;
};

extern void RasterTriangle(const cElTriangleComp & aTri,std::vector<cSegEntierHor> & aRes);



#endif //  _ELISE_INCLUDE_GENERAL_PTXD_H_



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
