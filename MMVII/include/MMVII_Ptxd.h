#ifndef  _MMVII_Ptxd_H_
#define  _MMVII_Ptxd_H_
namespace MMVII
{


/** \file MMVII_Ptxd.h
    \brief Basic N-dimensionnal point facilities

   Don't know exactly where we go ...  Probably will contain :

      - point in N dim template
      - specialization to 1,2,3,4  dims
      - dynamic dyn
      - boxes on others "small" connected classes

*/


template <class Type,const int Dim> class cPtxd;


/*   Relation/Bijection entre cPtxd et cPt1d,cPt2d etc ....
   La conversion de fait dans les deux sens

      cPtxd<Type,1> => cPt1d via le constructeur
      cPt1d =>  cPtxd<Type,1>  implicite via l'heritage

      Les deux sont donc plus ou moins synomymes 
*/


///  template class for Points
/**
   This class allow to manipulate points indepently of their dimension,
   with no or few, waste of performance
*/

template <class Type,const int Dim> class cPtxd
{
    public :
       /// Maybe some function will require generic access to data
       Type * PtRawData() {return mCoords;}
       const Type * PtRawData() const {return mCoords;}

       Type & operator[] (int aK) 
       {
          MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<Dim),"Bad point access");
          return mCoords[aK];
       }
       const Type & operator[] (int aK)  const
       {
          MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<Dim),"Bad point access");
          return mCoords[aK];
       }

       // cPtxd(const cPt1d<Type>);
       // cPtxd(const cPt2d<Type>);
         // MMVII_INTERNAL_ASSERT(Dim==1);

       /// Some function requires default constructor (serialization ?)
       cPtxd() {}

       /// Some function requires default constructor (serialization ?)
       static cPtxd<Type,Dim>  PCste(const Type & aVal) 
       { 
           cPtxd<Type,Dim> aRes;
           for (int aK=0 ; aK<Dim; aK++) 
               aRes.mCoords[aK]= aVal;
           return aRes;
       }
       /// Contructor for 1 dim point, statically checked
       cPtxd(const Type & x) :  mCoords{x} {static_assert(Dim==1,"bad dim in cPtxd initializer");}
       /// Contructor for 2 dim point, statically checked
       cPtxd(const Type & x,const Type &y) :  mCoords{x,y} {static_assert(Dim==2,"bad dim in cPtxd initializer");}
       /// Contructor for 3 dim point, statically checked
       cPtxd(const Type & x,const Type &y,const Type &z) :  mCoords{x,y,z} {static_assert(Dim==3,"bad dim in cPtxd initializer");}

        inline Type & x()             {static_assert(Dim>=1,"bad dim in cPtxd initializer");return mCoords[0];}
        inline const Type & x() const {static_assert(Dim>=1,"bad dim in cPtxd initializer");return mCoords[0];}

        inline Type & y()             {static_assert(Dim>=2,"bad dim in cPtxd initializer");return mCoords[1];}
        inline const Type & y() const {static_assert(Dim>=2,"bad dim in cPtxd initializer");return mCoords[1];}

        inline Type & z()             {static_assert(Dim>=2,"bad dim in cPtxd initializer");return mCoords[2];}
        inline const Type & z() const {static_assert(Dim>=2,"bad dim in cPtxd initializer");return mCoords[2];}

    protected :
       Type mCoords[Dim];
};
/*
template <class Type> inline cPt1d<Type> operator * (const Type & aVal ,const cPt1d<Type> & aP) 
{return  cPt1d<Type>(aP.x()*aVal);}
template <class Type> inline cPt1d<Type> operator * (const cPt1d<Type> & aP,const Type & aVal) 
{return  cPt1d<Type>(aP.x()*aVal);}
template <class Type> inline cPt1d<Type> operator / (const cPt1d<Type> & aP,const Type & aVal) 
{return  cPt1d<Type>(aP.x()/aVal);}
template <class Type> inline bool operator == (const cPt1d<Type> & aP1,const cPt1d<Type> & aP2) {return  (aP1.x()==aP2.x());}
template <class Type> inline bool operator != (const cPt1d<Type> & aP1,const cPt1d<Type> & aP2) {return !(aP1==aP2);}
*/

//  operator +
template <class Type> inline cPtxd<Type,1> operator + (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{ return cPtxd<Type,1>(aP1.x() + aP2.x()); }
template <class Type> inline cPtxd<Type,2> operator + (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{ return cPtxd<Type,2>(aP1.x() + aP2.x(),aP1.y() + aP2.y()); }
template <class Type> inline cPtxd<Type,3> operator + (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{ return cPtxd<Type,3>(aP1.x() + aP2.x(),aP1.y() + aP2.y(),aP1.z()+aP2.z()); }

//  operator -
template <class Type> inline cPtxd<Type,1> operator - (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{ return cPtxd<Type,1>(aP1.x() - aP2.x()); }
template <class Type> inline cPtxd<Type,2> operator - (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{ return cPtxd<Type,2>(aP1.x() - aP2.x(),aP1.y() - aP2.y()); }
template <class Type> inline cPtxd<Type,3> operator - (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{ return cPtxd<Type,3>(aP1.x() - aP2.x(),aP1.y() - aP2.y(),aP1.z()-aP2.z()); }

template <class Type> inline cPtxd<Type,1> operator - (const cPtxd<Type,1> & aP) {return  cPtxd<Type,1>(-aP.x());}
template <class Type> inline cPtxd<Type,2> operator - (const cPtxd<Type,2> & aP) {return  cPtxd<Type,2>(-aP.x(),-aP.y());}
template <class Type> inline cPtxd<Type,3> operator - (const cPtxd<Type,3> & aP) {return  cPtxd<Type,3>(-aP.x(),-aP.y(),-aP.z());}


// operator *
template <class Type> inline cPtxd<Type,1> operator * (const Type & aVal ,const cPtxd<Type,1> & aP) 
{return  cPtxd<Type,1>(aP.x()*aVal);}
template <class Type> inline cPtxd<Type,1> operator * (const cPtxd<Type,1> & aP,const Type & aVal) 
{return  cPtxd<Type,1>(aP.x()*aVal);}
template <class Type> inline cPtxd<Type,2> operator * (const Type & aVal ,const cPtxd<Type,2> & aP) 
{return  cPtxd<Type,2>(aP.x()*aVal,aP.y()*aVal);}
template <class Type> inline cPtxd<Type,2> operator * (const cPtxd<Type,2> & aP,const Type & aVal) 
{return  cPtxd<Type,2>(aP.x()*aVal,aP.y()*aVal);}
template <class Type> inline cPtxd<Type,3> operator * (const Type & aVal ,const cPtxd<Type,3> & aP) 
{return  cPtxd<Type,3>(aP.x()*aVal,aP.y()*aVal,aP.z()*aVal);}
template <class Type> inline cPtxd<Type,3> operator * (const cPtxd<Type,3> & aP,const Type & aVal) 
{return  cPtxd<Type,3>(aP.x()*aVal,aP.y()*aVal,aP.z()*aVal);}


//  operator ==
template <class Type> inline bool operator == (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{return  (aP1.x()==aP2.x());}
template <class Type> inline bool operator == (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{return  (aP1.x()==aP2.x()) && (aP1.y()==aP2.y());}
template <class Type> inline bool operator == (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{return  (aP1.x()==aP2.x()) && (aP1.y()==aP2.y()) && (aP1.z()==aP2.z());}

//  operator !=
template <class Type> inline bool operator != (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{return  (aP1.x()!=aP2.x());}
template <class Type> inline bool operator != (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{return  (aP1.x()!=aP2.x()) || (aP1.y()!=aP2.y());}
template <class Type> inline bool operator != (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{return  (aP1.x()!=aP2.x()) || (aP1.y()!=aP2.y()) ||  (aP1.z()!=aP2.z());}

// Operator SupEq  :  P1.k() >= P2.k() for all coordinates
template <class Type> inline bool SupEq  (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{return  (aP1.x()>=aP2.x());}
template <class Type> inline bool SupEq  (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{return  (aP1.x()>=aP2.x()) && (aP1.y()>=aP2.y());}
template <class Type> inline bool SupEq  (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{return  (aP1.x()>=aP2.x()) && (aP1.y()>=aP2.y()) && (aP1.z()>=aP2.z());}


template <class Type> inline cPtxd<Type,1> PtSupEq  (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{ return cPtxd<Type,1> (std::max(aP1.x(),aP2.x())); }
template <class Type> inline cPtxd<Type,2> PtSupEq  (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{ return cPtxd<Type,2> (std::max(aP1.x(),aP2.x()),std::max(aP1.y(),aP2.y())); }
template <class Type> inline cPtxd<Type,3> PtSupEq  (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{ return cPtxd<Type,3> (std::max(aP1.x(),aP2.x()),std::max(aP1.y(),aP2.y()),std::max(aP1.z(),aP2.z())); }


// Operator InfStr  :  P1.k() < P2.k() for all coordinates
template <class Type> inline bool InfStr  (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{return  (aP1.x()<aP2.x());}
template <class Type> inline bool InfStr  (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{return  (aP1.x()<aP2.x()) && (aP1.y()<aP2.y());}
template <class Type> inline bool InfStr  (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{return  (aP1.x()<aP2.x()) && (aP1.y()<aP2.y()) && (aP1.z()<aP2.z());}

template <class Type> inline cPtxd<Type,1> PtInfStr  (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{ return cPtxd<Type,1> (std::min(aP1.x(),aP2.x()-1)); }
template <class Type> inline cPtxd<Type,2> PtInfStr  (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{ return cPtxd<Type,2> (std::min(aP1.x(),aP2.x()-1),std::min(aP1.y(),aP2.y()-1)); }
template <class Type> inline cPtxd<Type,3> PtInfStr  (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{ return cPtxd<Type,3> (std::min(aP1.x(),aP2.x()-1),std::min(aP1.y(),aP2.y()-1),std::min(aP1.z(),aP2.z()-1)); }


// Operator InfEq  :  P1.k() <= P2.k() for all coordinates
template <class Type> inline bool InfEq  (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2) 
{return  (aP1.x()<=aP2.x());}
template <class Type> inline bool InfEq  (const cPtxd<Type,2> & aP1,const cPtxd<Type,2> & aP2) 
{return  (aP1.x()<=aP2.x()) && (aP1.y()<=aP2.y());}
template <class Type> inline bool InfEq  (const cPtxd<Type,3> & aP1,const cPtxd<Type,3> & aP2) 
{return  (aP1.x()<=aP2.x()) && (aP1.y()<=aP2.y()) && (aP1.z()<=aP2.z());}




//  operator << 
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,1> &aP)
{ return  OS << "[" << aP.x() << "]"; }
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,2> &aP)
{ return  OS << "[" << aP.x() << "," << aP.y() << "]"; }
template <class Type> std::ostream & operator << (std::ostream & OS,const cPtxd<Type,3> &aP)
{ return  OS << "[" << aP.x() << "," << aP.y() << "," << aP.z()<< "]"; }

/* Possible generik version, usefull ? I doubt ....
template <class Type,const int aDim>
    inline cPtxd<Type,aDim> operator +
          (const cPtxd<Type,aDim> & aP1,const cPtxd<Type,aDim> & aP2)
{
    cPtxd<Type,aDim> aRes;
    for (int aK=0 ; aK<aDim ; aK++)
        aRes.Data()[aK] = aP1.Data()[aK] + aP2.Data()[aK];

    return aRes;
}
*/

    ///  1 dimension specializatio,
typedef cPtxd<double,1>  cPt1dr ;
typedef cPtxd<int,1>     cPt1di ;
typedef cPtxd<float,1>   cPt1df ;

    ///  2 dimension specialization
typedef cPtxd<double,2>  cPt2dr ;
typedef cPtxd<int,2>     cPt2di ;
typedef cPtxd<float,2>   cPt2df ;

    ///  3 dimension specialization
typedef cPtxd<double,3>  cPt3dr ;
typedef cPtxd<int,3>     cPt3di ;
typedef cPtxd<float,3>   cPt3df ;


};

#endif  //  _MMVII_Ptxd_H_
