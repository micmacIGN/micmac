#ifndef CMSDAPPLI_H
#define CMSDAPPLI_H

#include "msd.h"
#include "lab_header.h"
#include "../../TpMMPD/TiePByMesh/Detector.h"


// call the appli in similar way than Digeo and SIFT: mm3d MSD -i inputname -o output name
// 08/2018; jo lisein: I implement MSD from work of Chebbi ENSG student, but unfortunately this detector seems to be quite non-efficient.
// to improve if usefulness : border effet, orientation of pt pbl. for the moment, I keep only pt that have only one orientation

void Migrate2Lab2wallis(Tiff_Im &image, Im2D<U_INT1,INT> &Output);
void Migrate2Lab2wallis(Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &Output);
void wallis(Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &WallEqIm);
template <class Type, class TyBase>
void Resizeim(Im2D<Type,TyBase> & im, Im2D<Type,TyBase> & Out, Pt2dr Newsize);

class cMSD1Im
{
public:
    cMSD1Im(int argc,char ** argv);
    void MSDBanniere();
private:
    std::string mNameIm, mOut, mTmpDir;
    bool mDebug;
    cInterfChantierNameManipulateur * mICNM;

    Im2D_U_INT1 mIm;

    // MSD and MSD param
    MsdDetector msd;
    double mTh;
    int mPR,mSAR,mKNN,mNMS;
    int mSc;

    // Instantiate the detector MSD
    void initMSD()
    {
          msd.setNameIm(mNameIm.substr(9,mNameIm.size()-6));
          msd.setDebug(mDebug);
          msd.setDir(mTmpDir);
          // tuning: althoug in the MSD paper they recommand Patch of 7 and SAR of 11, which is huge, a small patch and search area is balanced by the multiscale approach.

          //the size of the patches under comparison
          msd.setPatchRadius(mPR);
          //the size of the area from which the patches to be compared are
          msd.setSearchAreaRadius(mSAR);
          //Non-Maxima Suppression: 5x5 (=radius of 2) may be sufficient and keep more point of course.
          msd.setNMSRadius(mNMS);
          // local maximum on saliency map are detected with size NMSRadius. in addition, this size enable to merge kp of different resolution
          // if NMSscaleR is true, cause bug!!
          msd.setNMSScaleRadius(0);
          msd.setThSaliency(mTh); // lower Threshold Saliency give more MSD point. 0.01 is ok with use of SFS filter, but lower value are ok too
          msd.setKNN(mKNN); // higher KNN give me more MSD points
          // change of scale between two pyramid layer, 1.25 by default is not enough at my taste
          msd.setScaleFactor(1.5);
          msd.setNScales(mSc);// -1 means all scale

          if(!ELISE_fp::IsDirectory(mTmpDir) && mDebug)
          {
              std::cout << "Create directory " << mTmpDir << "\n";
              ELISE_fp::MkDir(mTmpDir);
          }
    }
};

#endif // CMSDAPPLI_H
