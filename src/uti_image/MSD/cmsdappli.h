#ifndef CMSDAPPLI_H
#define CMSDAPPLI_H

#include "msd.h"
#include "lab_header.h"
extern void self_match_lebris( list<DigeoPoint> &i_list,
                        double _min_dist,  	// min distance of second best point to avoid abiguity
                        int i_nbMaxPriPoints = SIFT_ANN_DEFAULT_MAX_PRI_POINTS );
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
    cMSD1Im(std::string aInputIm,std::string aOutTP,bool aDebug,int aPR,int aSAR,double aTh,int aNMS,int aKNN,double aSc,std::string aDir);
    void MSDBanniere();
private:
    std::string mNameIm1, mOut, mTmpDir;
    bool mDebug;
    cInterfChantierNameManipulateur * mICNM;
    MsdDetector msd;
    Im2D_U_INT1 mIm1;

    void initMSD()
    {
        // Instantiate the detector MSD
          msd.setDebug(mDebug);
          //the size of the patches under comparison
          msd.setPatchRadius(3);
          //the size of the area from which the patches to be compared are
          msd.setSearchAreaRadius(3);
          //Non-Maxima Suppression: 5x5 (=radius of 2) may be sufficient and keep more point of course.
          msd.setNMSRadius(3);
          // tuning: althoug in the MSD paper they recommand Patch of 7 and SAR of 11, which is huge, a small patch and search area is balanced by the multiscale approach.
          // local maximum on saliency map are detected with size NMSRadius. in addition, this size enable to merge kp of different resolution

          // if NMSscaleR is true, cause bug!!
          msd.setNMSScaleRadius(0);
          msd.setThSaliency(0.01); // lower Threshold Saliency give more MSD point. 0.01 is ok with use of SFS filter
          msd.setKNN(5); // higher KNN give me more MSD points
          // change of scale between two pyramid layer, 1.25 by default is not enough at my taste
          msd.setScaleFactor(1.5);
          msd.setNScales(-1);// -1 means all scale
          // orientation computed in MSD code is based on gray image, not on gradient. I use Digeo method implemented here to compute orientation on gradient
          // msd.setComputeOrientation(false);
          msd.setCircularWindow(false); // circular w reduce the number of MSD. Slow down the process considerably
          msd.setRefinedKP(false);
    }
};

#endif // CMSDAPPLI_H
