#ifndef  _MMVII_Ptxd_H_
#define  _MMVII_Ptxd_H_


template <class Type,const int Dim> struct cPtxd
{
    public :
       Type * Data() {return mCoords;}
    protected :
       Type mCoords[Dim];
};

    // specialisations 1 D

template <class Type> class cPt1d : public cPtxd<Type,1>
{
     public :
        // typedef typename cPtxd<Type,1> tBase;
        typedef cPtxd<Type,1> tBase;

        cPt1d(const Type& anX) {tBase::mCoords[0] = anX;}

        Type & x()             {return tBase::mCoords[0];}
        const Type & x() const {return tBase::mCoords[0];}
};

    // specialisations 2 D

template <class Type> class cPt2d : public cPtxd<Type,2>
{
     public :
        typedef cPtxd<Type,2> tBase;

        Type & x()             {return tBase::mCoords[0];}
        const Type & x() const {return tBase::mCoords[0];}
        Type & y()             {return tBase::mCoords[1];}
        const Type & y() const {return tBase::mCoords[1];}
};



/*
template <class Type,1> operator + (const cPtxd<Type,1> & aP1,const cPtxd<Type,1> & aP2)
{
    cPtxd<Type,1> aRes;
    aRes.mCoords[0]
}
*/

#endif  //  _MMVII_Ptxd_H_
