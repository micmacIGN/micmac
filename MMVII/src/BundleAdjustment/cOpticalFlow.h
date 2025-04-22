#include "MMVII_Interpolators.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Geom2D.h"
#include "Eigen/Sparse"

using namespace Eigen;

namespace MMVII
{

template <class TypeIm> class cOpticalFlow
{
public:
    typedef tREAL8 tTypeDisp;
    typedef cIm2D<tTypeDisp> tImDispl;
    typedef cDataIm2D<tTypeDisp> tDImDispl;
    typedef cIm2D<tU_INT1> tImMasq;
    typedef cDataIm2D<tU_INT1> tDImMasq;
    
    cOpticalFlow(const cIm2D<TypeIm> & aIm1,
                 const cIm2D<TypeIm> & aIm2,
                 tImMasq & aImMasq1);
    cOpticalFlow(const std::string & ,
                 const std::string & ,
                 const  std::string & );
    void udpateDispl(tDImDispl & anActDispX,
                     tDImDispl & anActDispY,
                     Eigen::SparseMatrix<tREAL8> & aSol);
    void udpateDisplDirect(tDImDispl & anActDispX,
                     tDImDispl & anActDispY);
    void refreshDisp(tDImDispl & anActDispX,
                     tDImDispl & anActDispY);
    void InitMat(bool isA);
    std::pair<tREAL8,tREAL8> diff(tDImDispl & anActualDisplX,tDImDispl & anActualDisplY);
    void SolveDisp(std::string & aNameOut);
    void SolveDispDirect(std::string & aNameOut, tREAL8 * TRANSFORM);
    void saveFlow(std::string & aNameOut, tREAL8 * transform);
    tDImDispl * DDispY() {return mDDispY;}
    tDImDispl * DDispX() {return mDDispX;}

    ~cOpticalFlow();
    
private:
    tImDispl mDispX;
    tDImDispl * mDDispX;
    tImDispl mDispY;
    tDImDispl * mDDispY;
    
    tImDispl mImGradX;
    tDImDispl * mDImGradX;
    
    tImDispl mImGradY;
    tDImDispl * mDImGradY;
    
    tImDispl mImGradt;
    tDImDispl * mDImGradt;
    tDImMasq * mDMasq1;
    
    // balance parameter lambda
    tREAL8 mLamda=0.0005;
    cDiffInterpolator1D  * mInterp=cDiffInterpolator1D::AllocFromNames({"Scale","2","100","Cubic","-0.5"});
    Eigen::SparseMatrix<tREAL8> mmatA=SparseMatrix<tREAL8>();
    Eigen::SparseMatrix<tREAL8> mmatB=SparseMatrix<tREAL8>();
};


};
