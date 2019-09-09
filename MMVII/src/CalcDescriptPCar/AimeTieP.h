#ifndef _AIME_TIEP_H_
#define _AIME_TIEP_H_

namespace MMVII
{

/// Proto-class for Aime TieP

/**  This very basic class is made to export result of MMV2 in a way they can
     be inspected in MMV1 (as MMV2 will not have tools equivalent to X11-MMV1 before a long)
     Certainly Aime TieP will be much more complex than this proto class
*/
class cProtoAimeTieP
{
    public :
        cProtoAimeTieP(const cPt2dr & aP,int aNumOct,int aNumIm,float aScaleInO,float aScaleAbs);
        void SetPt(const cPt2dr & );
        const cPt2dr & Pt() const;
        int   NumOct() const;
        int   NumIm() const;
        float ScaleInO() const;
        float ScaleAbs() const;

    private :
        cPt2dr mPt;
        int    mNumOct;
        int    mNumIm;
        float  mScaleInO;
        float  mScaleAbs;
};

/// Interface class for export Proto Aime

/**  As I want to maintain the principle that only a very minimum
   of code of MMV2 communicate with MMV1, this interface class was created
*/

template <class Type> class cInterf_ExportAimeTiep : public cMemCheck
{
     public :
         static cInterf_ExportAimeTiep<Type> * Alloc(bool IsMin,int ATypePt,const std::string & aName,bool ForInspect );
         virtual ~cInterf_ExportAimeTiep();
         virtual void AddAimeTieP(const cProtoAimeTieP & aPATP ) = 0;
         virtual void Export(const std::string &) = 0;
         virtual void SetCurImages(cIm2D<Type> Im0,cIm2D<Type> ImCarac,double aScalInO) = 0;
       
     protected :

};



};

#endif // _AIME_TIEP_H_

